import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import cosine_similarity
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import pandas as pd
from peft import LoraConfig, get_peft_model
import argparse
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

class DocumentPairDataset(Dataset):
    def __init__(self, data, tokenizer, model_name):
        self.data = data
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = 512  # Default max length

        if model_name == "longformer":
            self.max_length = 1024

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized_text1 = self.tokenizer(item['text1'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        tokenized_text2 = self.tokenizer(item['text2'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        
        if self.model_name == "rrivera1849/LUAR-MUD":
            # tokenized_text1['input_ids'] = tokenized_text1['input_ids'].reshape(1, 1, -1).to(device)
            # tokenized_text1['attention_mask'] = tokenized_text1['attention_mask'].reshape(1, 1, -1).to(device)
            context = {
                "tokenized_text1": tokenized_text1,
                "tokenized_text2": tokenized_text2
            }
        else:
            context = {
                    "input_ids1": tokenized_text1['input_ids'].squeeze(0),
                    "attention_mask1": tokenized_text1['attention_mask'].squeeze(0),
                    "input_ids2": tokenized_text2['input_ids'].squeeze(0),
                    "attention_mask2": tokenized_text2['attention_mask'].squeeze(0)
                }
            
        label = torch.tensor(item["label"])
        return {"context": context, "labels": label}
    
def process_posts(post_df):
    data = []
    for i, row in tqdm(post_df.iterrows(), total=post_df.shape[0], desc="Processing Posts"):
        data.append({"text1":row['post_1'],
                        "text2":row['post_2'],
                        "label":int(row['same'])})
    
    return data

class SiameseRoberta(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model_name = model
        self.enc_model = AutoModel.from_pretrained(model, trust_remote_code=True)
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['query', 'value'],
            lora_dropout=0.05,
            bias="none",
        )
        self.enc_model = get_peft_model(self.enc_model, config)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        if self.model_name == "rrivera1849/LUAR-MUD":
            output1 = self.enc_model(**input_ids1) 
            output2 = self.enc_model(**attention_mask1)
        else:
            output1 = self.enc_model(input_ids1, attention_mask=attention_mask1)[1]  # pooler_output
            output2 = self.enc_model(input_ids2, attention_mask=attention_mask2)[1]  # pooler_output
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos_sim = cosine_similarity(output1, output2)
        loss = torch.mean((1-label) * 0.5 * torch.pow(cos_sim, 2) +
                          label * 0.5 * torch.pow(torch.clamp(self.margin - cos_sim, min=0.0), 2))
        return loss
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='roberta-base', choices=['roberta-base', 'roberta-large', 'longformer', 'luar'])
    parser.add_argument('-d', '--dataset', type=str, default='reddit', choices=['reddit', 'amazon', 'fanfiction'])
    args = parser.parse_args()
    
    loss = ContrastiveLoss()

    train_df = pd.read_csv(f"../data/{args.dataset}/train.csv")
    dev_df = pd.read_csv(f"../data/{args.dataset}/dev.csv")
    test_df = pd.read_csv("../data/reddit/test.csv")

    if args.model == "longformer":
        model_name = "allenai/longformer-4096"
    elif args.model == "luar":
        model_name = 'rrivera1849/LUAR-MUD'
    else:
        model_name = args.model
        
    model = SiameseRoberta(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    train_data = process_posts(train_df)
    dev_data = process_posts(dev_df)

    train_dataset = DocumentPairDataset(train_data, tokenizer, model_name)
    val_dataset = DocumentPairDataset(dev_data, tokenizer, model_name)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_fn = ContrastiveLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    best_val_loss = float('inf')
    num_epochs = 5

    # Initialize GradScaler
    scaler = GradScaler()

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            context = batch['context']
            labels = batch['labels'].to(device)
            
            if model_name == "rrivera1849/LUAR-MUD":
                with autocast():
                    output1, output2 = model(context['tokenized_text1'].to(device), context['tokenized_text2'].to(device), input_ids2 = None, attention_mask2 = None)
                loss = loss_fn(output1, output2, labels)
            else:
                with autocast():
                    output1, output2 = model(context['input_ids1'].to(device), context['attention_mask1'].to(device),
                                         context['input_ids2'].to(device), context['attention_mask2'].to(device))
                loss = loss_fn(output1, output2, labels)
            
            # Scales the loss, and calls backward() to create scaled gradients
            scaler.scale(loss).backward()
            # Unscales gradients and calls optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
            
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                context = batch['context']
                labels = batch['labels'].to(device)
                if model_name == "rrivera1849/LUAR-MUD":
                    with autocast():
                        output1, output2 = model(context['tokenized_text1'].to(device), context['tokenized_text2'].to(device), input_ids2 = None, attention_mask2 = None)
                    loss = loss_fn(output1, output2, labels)
                else:
                    with autocast():
                        output1, output2 = model(context['input_ids1'].to(device), context['attention_mask1'].to(device),
                                         context['input_ids2'].to(device), context['attention_mask2'].to(device))
                loss = loss_fn(output1, output2, labels)
                
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'../model/contrastive_loss_finetuned_{args.model}_{args.dataset}.pt')
            print("Model saved as validation loss improved.")
    
    vector_map = pd.read_pickle(f'{args.dataset}_vector_map.pkl')
    
    post1s = list(test_df['post_1'])
    post2s = list(test_df['post_2'])

    model.eval()  # Set the model to evaluation mode

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Function to get embeddings
    def get_embeddings(text, model_name):
        tokenized = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        if model_name == "rrivera1849/LUAR-MUD":
            tokenized['input_ids'] = tokenized['input_ids'].reshape(1, 1, -1).to(device)
            tokenized['attention_mask'] = tokenized['attention_mask'].reshape(1, 1, -1).to(device)
            with torch.no_grad():
                outputs = model(tokenized, tokenized, input_ids2=None, attention_mask2=None)
            embedding = outputs[0].cpu()
        else:
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)

            with torch.no_grad():
                # You can use either output1 or output2 since both are the same model and weights
                embedding = model.enc_model(input_ids, attention_mask=attention_mask)[1].cpu()  # pooler_output
        return embedding

    gram2vec_cosims = []
    from sklearn.metrics.pairwise import cosine_similarity

    for i, row in test_df.iterrows():
        vector1 = vector_map[row['post_1_id']]
        vector2 = vector_map[row['post_2_id']]
        cosim = cosine_similarity(vector1, vector2)
        gram2vec_cosims.append(cosim[0][0])

    neural_cosims = []
    for i, post in tqdm(enumerate(post1s), desc="Processing posts for embeddings", total=len(post1s)):
        post1 = post1s[i]
        post2 = post2s[i]

        embedding1 = get_embeddings(post1, model_name)
        embedding2 = get_embeddings(post2, model_name)
        neural_cosim = cosine_similarity(embedding1, embedding2)
        neural_cosims.append(neural_cosim[0][0])    

    weights = np.linspace(0, 1, 101)  # Test weights from 0 to 1 in steps of 0.01
    best_weight = 0
    best_auc = 0
    
    true_labels = list(test_df['same'])

    for weight in weights:
        combined_cosims = weight * np.array(gram2vec_cosims) + (1 - weight) * np.array(neural_cosims)
        fpr, tpr, _ = roc_curve(true_labels, combined_cosims)
        auc_score = auc(fpr, tpr)
        if auc_score > best_auc:
            best_auc = auc_score
            best_weight = weight

    print(f"Best weight: {best_weight}, Best AUC: {best_auc:.3f}")

    # Plot the ROC curve for the best weight
    combined_cosims = best_weight * np.array(gram2vec_cosims) + (1 - best_weight) * np.array(neural_cosims)
    gram2vec_fpr, gram2vec_tpr, _ = roc_curve(true_labels, gram2vec_cosims)
    neural_fpr, neural_tpr, _ = roc_curve(true_labels, neural_cosims)
    fpr, tpr, _ = roc_curve(true_labels, combined_cosims)
    gram2vec_auc = auc(gram2vec_fpr, gram2vec_tpr)
    neural_auc = auc(neural_fpr, neural_tpr)
    auc_score = auc(fpr, tpr)
    plt.figure()
    plt.text(0.05, 0.95, f'Best Weight: {best_weight:.3f}', fontsize=12, fontweight='bold', ha='left', va='top', color='red', bbox=dict(facecolor='white', alpha=0.8))

    # plt.text(0.5, 0.01, f'Best Weight: {f1_best_weight:.3f}, Best F1: {best_f1:.3f}, Threshold: {best_threshold:.2f}', fontsize=12, ha='center', color='red', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble AUC = {auc_score:.3f}')
    plt.plot(gram2vec_fpr, gram2vec_tpr, color='blue', lw=2, label=f'gram2vec AUC = {gram2vec_auc:.3f}')
    plt.plot(neural_fpr, neural_tpr, color='green', lw=2, label=f'neural AUC = {neural_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"../results/neural_cosine/graphs/neural_cosine_{args.model}_{args.dataset}_auc.png")