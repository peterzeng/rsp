import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.utils.data import Dataset
import argparse
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import autocast
from explainable_module import Gram2VecModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AuthorClassificationModel(torch.nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type
        if model_type == "luar":
            self.enc_model = AutoModel.from_pretrained('rrivera1849/LUAR-MUD', trust_remote_code=True)
        elif model_type == "longformer":
            self.enc_model = AutoModel.from_pretrained('allenai/longformer-base-4096')
        elif model_type == "roberta":
            self.enc_model = AutoModel.from_pretrained('roberta-base')
        elif model_type == "roberta-large":
            self.enc_model = AutoModel.from_pretrained('roberta-large')
        else:
            raise ValueError("Unsupported model type specified.")
        
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query","value"],
            lora_dropout=0.05,
            bias="none",
        )
        self.enc_model = get_peft_model(self.enc_model, config)
        
        if model_type == "luar":
            self.classifier = torch.nn.Linear(512, 2)
        elif model_type == "longformer" or model_type == "roberta":
            self.classifier = torch.nn.Linear(768, 2)
        elif model_type == "roberta-large":
            self.classifier = torch.nn.Linear(1024, 2)
        else:
            raise ValueError("Unsupported model type specified.")
        
    def forward(self, context, labels=None):
        if self.model_type in ['roberta', 'roberta-large', 'longformer']:
            outputs = self.enc_model(**context, output_hidden_states=True).hidden_states[-1]
            embeds = outputs[:, 0, :]
            
        elif self.model_type == 'luar':
            outputs = self.enc_model(**context)
            embeds = outputs

        logits = self.classifier(embeds)
        logits = logits.to(torch.float32)
        # print("Logits type:", type(logits))
        # print("Logits shape:", logits.shape)
        # if labels is not None:
        #     print("Labels type:", type(labels))
        #     print("Labels shape:", labels.shape)
        if labels is not None:
            labels = labels.to(torch.long)
            loss_fct = torch.nn.CrossEntropyLoss()  # Loss function for classification
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return {
                "loss": loss,
                "logits": logits
            }
        return logits

class DocumentPairDataset(Dataset):
    def __init__(self, data, model_type):
        self.data = data

        if model_type == "longformer":
            self.model_type = "allenai/longformer-base-4096"
        elif model_type == "roberta":
            self.model_type = "roberta-base"
        elif model_type == "roberta-large":
            self.model_type = "roberta-large"
        elif model_type == "luar":
            self.model_type = "rrivera1849/LUAR-MUD"
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.max_length = 1024 if model_type == 'longformer' else 512
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = self.tokenizer(item['text1'], item['text2'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        
        if not self.model_type == "rrivera1849/LUAR-MUD":
            context = {key: val[0] for key, val in context.items()}  # Remove the extra batch dimension

        label = torch.tensor(item["residual"], dtype=torch.long)
        return {"context": context, "labels": label}

def process_posts(post_df, g2v_vectorizer):
    data = []
    for i, row in tqdm(post_df.iterrows(), total=post_df.shape[0], desc="Processing Posts"):
        _, _, cosim = g2v_vectorizer.get_vector_and_score(row['post_1'], row['post_2'], row['post_1_id'], row['post_2_id'])
        residual = row['same'] - cosim
        data.append({"text1":row['post_1'],
                        "text2":row['post_2'],
                        "residual":residual})
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on residual data.")
    parser.add_argument("-m", "--model_type", type=str, default="roberta", choices=["longformer", "roberta", "roberta-large", "luar"], help="Type of model to use for training.")
    parser.add_argument("-d", "--dataset", type=str, default="reddit", choices=['reddit', 'amazon', 'fanfiction'], help="Dataset to use for training.")
    parser.add_argument("-r", "--run_id", type=str, default="0", help="Run ID for the experiment.")
    args = parser.parse_args()

    model = AuthorClassificationModel(args.model_type)

    train_df = pd.read_csv(f"../data/{args.dataset}/train.csv")
    dev_df = pd.read_csv(f"../data/{args.dataset}/dev.csv")
    test_df = pd.read_csv(f"../data/{args.dataset}/test.csv")

    g2v_vectorizer = Gram2VecModule(dataset=args.dataset)
    train_data = process_posts(train_df, g2v_vectorizer)
    dev_data = process_posts(dev_df, g2v_vectorizer)
    test_data = process_posts(test_df, g2v_vectorizer)
    
    train_dataset = DocumentPairDataset(train_data, args.model_type)
    dev_dataset = DocumentPairDataset(dev_data, args.model_type)
    test_dataset = DocumentPairDataset(test_data, args.model_type)
    
    # Optimized for A6000 GPU
    if args.model_type == 'roberta-large' or args.model_type == 'longformer':
        b_size = 32
    elif args.model_type == 'roberta' or args.model_type == 'style':
        b_size = 64
    elif args.model_type == 'luar':
        b_size = 128

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=b_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=b_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    model.to(device)
    num_epochs = 10
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    best_val_loss = float('inf') 
    early_stopping_counter = 0
    early_stopping_patience = 3
    
 
    accumulation_steps = 1 
    print(f"number of accuulation steps: {accumulation_steps}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dataloader = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", position=1, leave=True)
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            with autocast():
                context = {k: v.to(device) for k, v in batch["context"].items()}
                labels = batch["labels"].to(device).to(torch.long) 
                outputs = model(context, labels=labels)
                loss = outputs["loss"] / accumulation_steps

            loss.backward()  

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()  
                optimizer.zero_grad()  

            train_loss += loss.item() * accumulation_steps 
            train_dataloader.set_postfix(loss=(train_loss / (i + 1)), refresh=False)

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")

        dev_dataloader = tqdm(dev_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", position=1, leave=True)
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dev_dataloader:
                with autocast():
                    context = {k: v.to(device) for k, v in batch["context"].items()}
                    labels = batch["labels"].to(device).to(torch.long) 
                    outputs = model(context, labels=labels)
                    loss = outputs["loss"]
                
                val_loss += loss.item()

        avg_val_loss = val_loss / len(dev_dataloader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")

        # Check if the current validation loss is the best we've seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model
            torch.save(model.state_dict(), f'../model/{args.model_type}_{args.dataset}_{args.run_id}_classification_baseline.pt')
            print(f"New best validation loss: {best_val_loss}. Model saved!")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Testing loop with a summary of per class F1, precision, and recall
    model.eval()
    all_labels = []
    all_predictions = []
    test_dataloader = tqdm(test_dataloader, desc=f"Testing", position=1, leave=True)
    with torch.no_grad():
        for batch in test_dataloader:
            context = {k: v.to(device) for k, v in batch["context"].items()}
            outputs = model(context)
            logits = outputs
            predictions = torch.argmax(logits, dim=-1)
            labels = batch['labels'].to(device)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # Calculate metrics per class
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    # Calculate precision, recall, f1, and support for each class without averaging
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)

    # Print per class results
    print("Class-wise metrics:")
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")

    # Calculate and print total metrics using 'macro' average
    total_precision, total_recall, total_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=0)
    print(f"Total Precision: {total_precision:.4f}")
    print(f"Total Recall: {total_recall:.4f}")
    print(f"Total F1 Score: {total_f1:.4f}")

    # Calculate and print total accuracy
    total_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Total Accuracy: {total_accuracy:.4f}")

    # Calculate and print accuracy for each class
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        class_mask = (all_labels == label)
        class_accuracy = accuracy_score(np.array(all_labels)[class_mask], np.array(all_predictions)[class_mask])
        print(f"Class {label} Accuracy: {class_accuracy:.4f}")
    
    import pandas as pd

    # Create a DataFrame to store the metrics
    metrics_data = {
        "Class": [f"Class {i}" for i in range(len(precision))],
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy": [accuracy_score(np.array(all_labels)[all_labels == i], np.array(all_predictions)[all_labels == i]) for i in np.unique(all_labels)]
    }

    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Save the DataFrame to a CSV file
    metrics_df.to_csv(f"../results/classification/{args.model_type}_{args.dataset}_{args.run_id}_class_metrics.csv", index=False)
    print("Metrics saved to class_metrics.csv")