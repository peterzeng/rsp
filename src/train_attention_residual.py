import torch
from torch.utils.data import DataLoader
from AttentionResidualModel import AttentionResidualDataset, AttentionResidualModel
import pandas as pd
from torch.optim import AdamW
from transformers import AutoModel
from tqdm.auto import tqdm
import logging
from sklearn.metrics.pairwise import cosine_similarity
from torch.cuda.amp import autocast
from sklearn.metrics import roc_curve, auc
import argparse
import os  
import csv
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def process_posts(post_df, g2v_vectorizer, normalized=False):
    data = []
    for i, row in tqdm(post_df.iterrows(), total=post_df.shape[0], desc="Processing Posts"):
        features1, features2, cosim = g2v_vectorizer.get_vector_and_score(row['document1'], row['document2'], row['doc_id1'], row['doc_id2'], normalized=normalized)
        residual = (1 - cosim) if row['same'] == 1 else (-1 - cosim)
        data.append({"text1":row['document1'],
                        "text2":row['document2'],
                        "features1":features1,
                        "features2":features2,
                        "residual":residual})
    
    return data

def generate_neural_feature_map(dataloader, model_type):
    if model_type == "/home/zengpe/rsp/LUAR-RU":
        model = AutoModel.from_pretrained(model_type, local_files_only=True, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_type, trust_remote_code=True)

    model.to(device)
    neural_cosims = []
    model.eval()
    with torch.no_grad():
        dataloader = tqdm(dataloader, desc="test loop", position=1 ,leave=True)

        for batch in dataloader:
            with autocast():
                doc1 = batch['text1']
                doc2 = batch['text2']
                doc1 = {k: v.to(device) for k, v in doc1.items()}
                doc2 = {k: v.to(device) for k, v in doc2.items()}
                outputs1 = model(**doc1).detach().cpu().numpy()
                outputs2 = model(**doc2).detach().cpu().numpy()
                
            cosims = cosine_similarity(outputs1, outputs2)
            neural_cosims.extend(cosims.diagonal())

    model.cpu()
    del model
    torch.cuda.empty_cache()
    return neural_cosims

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on residual data.")
    parser.add_argument("-m", "--model_type", type=str, default="luar", choices=["roberta", "roberta-large", "luar", "style", "luar-ru"], help="Type of model to use for training.")
    parser.add_argument("-d", "--dataset", type=str, default="reddit", choices=["fanfiction", "reddit", "amazon", "hiatus", "imbalanced_reddit","hiatus_combined"], help="Dataset to use for training.")
    parser.add_argument("-r", "--run_id", type=str, default="", help="Run ID for the experiment.")
    parser.add_argument("-p", "--percentage", type=float, default=1, help="Percentage of data to sample for training, validation, and testing.")
    parser.add_argument("-c", "--config", type=str, default="config.txt", help="Path to the config file.")
    parser.add_argument("-s", "--save_dir", type=str, default="vector_cache", help="Path to the save directory.")
    parser.add_argument("-f", "--feature_dim", type=int, default=617, help="Feature dimension.")
    parser.add_argument("-n", "--normalized", type=bool, default=False, help="Whether to use normalized vectors.")
    parser.add_argument("-l", "--language", type=str, default="en", choices=["en", "ru"], help="Language of the dataset.")
    args = parser.parse_args()

    vectorizer_configs = {
        "pos_unigrams":1,
        "pos_bigrams":1,
        "func_words":1,
        "punctuation":1,
        "letters":0,
        "emojis":1,
        "dep_labels":1, 
        "morph_tags":1,
        "sentences":1,
        "num_tokens":0 
    }

    if args.language == "en":
        train_df = pd.read_csv(f"../data/{args.dataset}/train.csv", encoding="utf-8")
        # train_df = pd.read_csv("../data/hiatus_combined_hard_negative/train.csv", encoding="utf-8")
        dev_df = pd.read_csv(f"../data/{args.dataset}/dev.csv", encoding="utf-8")
        test_df = pd.read_csv(f"../data/{args.dataset}/test.csv", encoding="utf-8")
        os.environ["LANGUAGE"] = "en"
        os.environ["SPACY_MODEL"] = "en_core_web_lg"
    elif args.language == "ru":
        train_df = pd.read_csv(f"../data/{args.dataset}/russian/train.csv", encoding="utf-8")
        dev_df = pd.read_csv(f"../data/{args.dataset}/russian/dev.csv", encoding="utf-8")
        test_df = pd.read_csv(f"../data/{args.dataset}/russian/test.csv", encoding="utf-8")
        os.environ["LANGUAGE"] = "ru"
        os.environ["SPACY_MODEL"] = "ru_core_news_lg"
        args.feature_dim = 613
    
    from explainable_module import Gram2VecModule
    g2v_vectorizer = Gram2VecModule(filepath=f"vector_cache/{args.dataset}_{args.run_id}_vector_map.pkl", dataset=args.dataset, save_dir=args.save_dir, run_id=args.run_id, configs=vectorizer_configs)
    # if not os.path.exists(f"vector_cache/{args.dataset}_{args.run_id}_normalized_vector_map.pkl"):
        # g2v_vectorizer.normalize_cache()
        # g2v_vectorizer.save_cache(normalized=True)

    if args.model_type == "luar":
        model_type = "rrivera1849/LUAR-MUD"
    elif args.model_type == "luar-ru":
        model_type = "/home/zengpe/rsp/LUAR-RU"
    elif args.model_type == "roberta":
        model_type = "sentence-transformers/all-distilroberta-v1"
    elif args.model_type == "roberta-large":
        model_type = "sentence-transformers/all-roberta-large-v1"
    elif args.model_type == "style":
        model_type = "AnnaWegmann/Style-Embedding"

    model = AttentionResidualModel(model_type=model_type, feature_dim=args.feature_dim)

    model.to(device)
    
    if args.percentage != 1:
        train_df_sampled = train_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility
        dev_df_sampled = dev_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility
        test_df_sampled = test_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility

        train_data = process_posts(train_df_sampled, g2v_vectorizer)
        dev_data = process_posts(dev_df_sampled, g2v_vectorizer)
        test_data = process_posts(test_df_sampled, g2v_vectorizer)
    else:
        train_data = process_posts(train_df, g2v_vectorizer, normalized=args.normalized)
        dev_data = process_posts(dev_df, g2v_vectorizer, normalized=args.normalized)
        test_data = process_posts(test_df, g2v_vectorizer, normalized=args.normalized)
        test_df_sampled = test_df

    if args.percentage == 1:
        # g2v_vectorizer.normalize_cache()
        g2v_vectorizer.save_cache()

    print(model_type)
    train_dataset = AttentionResidualDataset(train_data, model_type = model_type)
    dev_dataset = AttentionResidualDataset(dev_data, model_type = model_type)
    test_dataset = AttentionResidualDataset(test_data, model_type = model_type)

    # Optimized for A6000 GPU 
    if args.model_type == "roberta-large":
        b_size = 16
    elif args.model_type == "style":
        b_size = 32
    elif args.model_type == 'luar' or args.model_type == "roberta":
        b_size = 64
    
    accumulation_steps = 1
    print(f"Using batch size of {b_size}")
    print(f"number of accuulation steps: {accumulation_steps}")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=b_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=b_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=.0001)
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 3

    # Define the total number of epochs
    num_epochs = 10
    
    pbar = tqdm(total=num_epochs, desc="Overall Training Progress", position=0)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_dataloader = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", position=1, leave=True)
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            with autocast():
                doc1 = batch['text1']
                doc2 = batch['text2']
                features1 = batch['features1']
                features2 = batch['features2']
                labels = batch["labels"].to(device)  # Move labels to the device
                outputs = model(doc1, doc2, features1, features2, labels=labels)
                # attention_weights = outputs["attention_weights"]
                # avg_attention = torch.mean(attention_weights, dim=1)
                # print(avg_attention)
                loss = outputs["loss"] / accumulation_steps  # Normalize loss to account for accumulation
            
            # print(outputs['logits'])
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()  # Update parameters
                optimizer.zero_grad()  # Reset gradients for the next set of accumulation steps

            # Detach the loss when accumulating
            total_loss += loss.detach().item() * accumulation_steps
            train_dataloader.set_postfix(loss=(total_loss / (i + 1)), refresh=False)
            # optimizer.step()

        avg_loss = total_loss/len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")

        model.eval()
        val_loss = 0
        val_labels = []
        val_predictions = []
        attention_weights_sum = None
        attention_weights_count = 0

        with torch.no_grad():
            dev_dataloader = tqdm(dev_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", position=1, leave=True)

            for batch in dev_dataloader:
                with autocast():
                    doc1 = batch['text1']
                    doc2 = batch['text2']
                    features1 = batch['features1']
                    features2 = batch['features2']
                    labels = batch["labels"].to(device)
                    outputs = model(doc1, doc2, features1, features2, labels=labels)
                    loss = outputs["loss"]
                
                val_loss += loss.item()
                predictions = outputs["logits"].squeeze().detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                val_labels.extend(labels)
                val_predictions.extend(predictions)

                # Accumulate attention weights
                batch_attention = outputs["attention_weights"].mean(dim=1)  # Average across heads
                if attention_weights_sum is None:
                    attention_weights_sum = batch_attention.sum(dim=0).cpu().numpy()
                else:
                    attention_weights_sum += batch_attention.sum(dim=0).cpu().numpy()
                attention_weights_count += batch_attention.shape[0]

        avg_val_loss = val_loss / len(dev_dataloader)
        print(f"Validation Loss: {avg_val_loss}")

        # Calculate and print average attention weights
        if attention_weights_count > 0:
            avg_attention_weights = attention_weights_sum / attention_weights_count
            print("\nAverage Attention Weights:")
            print(avg_attention_weights)
            
            print("\nAttention Weight Interpretations:")
            input_names = ['hidden1', 'hidden2', 'features1', 'features2']
       
            for i, name in enumerate(input_names):
                print(f"Attention weight for {name}: {avg_attention_weights[i]:.4f}")
                
            # else:
            #     print("Unexpected shape of attention weights. Please check the model output.")
            #  if len(avg_attention_weights.shape) == 2:
            #     for i, source in enumerate(input_names):
            #         for j, target in enumerate(input_names):
            #             print(f"{source} attending to {target}: {avg_attention_weights[i, j]:.4f}")
            # elif len(avg_attention_weights.shape) == 1:
        logging.info(f'Epoch {epoch+1}/{num_epochs} completed.')
        # Save the model if the validation loss is the best we've seen so far.
        if avg_val_loss < best_val_loss:
            print("saving model")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"../model/{args.model_type}_{args.dataset}_{args.run_id}_attention_residual.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                }, f"../model/{args.model_type}_{args.dataset}_{args.run_id}_attention_residual_checkpoint.pt")
            early_stopping_counter = 0  # reset counter after improvement
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        pbar.update(1)

    pbar.close()

    neural_cosims = generate_neural_feature_map(test_dataloader, model_type)

    # TESTING LOOP TO SEE HOW MUCH FINETUNED MODEL CORRECTS GRAM2VEC:
    gram2vec_cosims = []
    vector_map = pd.read_pickle(f'{args.save_dir}/{args.dataset}_{args.run_id}_vector_map.pkl')

    for i, row in test_df_sampled.iterrows():
        vector1 = vector_map[row['doc_id1']]
        vector2 = vector_map[row['doc_id2']]
        cosim = cosine_similarity(vector1, vector2)
        gram2vec_cosims.append(cosim[0][0])

    predicted_labels = []

    model.eval()
    with torch.no_grad():
        test_dataloader = tqdm(test_dataloader, desc="test loop", position=1 ,leave=True)

        for batch in test_dataloader:
            # print(batch['labels'])
            with autocast():
                doc1 = batch['text1']
                doc2 = batch['text2']
                features1 = batch['features1']
                features2 = batch['features2']
                labels = batch["labels"].to(device)  # Move labels to the device
                outputs = model(doc1, doc2, features1, features2, labels=labels)
                # print(outputs)
            # print(outputs["logits"])
            predictions = outputs["logits"].squeeze().detach().cpu().numpy()  # Adjust based on your model's output
            predicted_labels.append(predictions)

    residual_cosims = [gram2vec_cosims[i] + predicted_labels[i] for i in range(len(predicted_labels))]
    same_labels = list(test_df_sampled['same'])
    
    ### EVALAUTION CODE ###
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for threshold in thresholds:
        g2v_correct = 0
        long_correct = 0
        true_labels = list(test_df_sampled['same'])

        for i in range(len(true_labels)):
            if true_labels[i] == 1:
                if (gram2vec_cosims[i] > threshold):
                    g2v_correct += 1
                if (residual_cosims[i] > threshold):
                    long_correct += 1
                    
            elif true_labels[i] == 0:
                if (gram2vec_cosims[i] < threshold):
                    g2v_correct += 1
                if (residual_cosims[i] < threshold):
                    long_correct += 1

        total = len(true_labels)
        # print(f"threshold: {threshold}, gram2vec correct: {g2v_correct}/{total}, residual correct: {long_correct}/{total}")
        # print(f"gram2vec accuracy: {g2v_correct/len(true_labels):.4f}, residual accuracy: {long_correct/len(true_labels):.4f}")
        from sklearn.metrics import precision_score, recall_score, f1_score

        # Convert lists to numpy arrays for compatibility with sklearn metrics
        true_labels_np = np.array(true_labels)
        predicted_labels_residual = np.array([1 if score > threshold else 0 for score in residual_cosims])
        # print(calculate_metrics(true_labels_np, predicted_labels_residual))
        predicted_labels_gram2vec = np.array([1 if score > threshold else 0 for score in gram2vec_cosims])

        gram2vec_f1_per_class = f1_score(true_labels_np, predicted_labels_gram2vec, average=None, zero_division=0)
        gram2vec_precision_per_class = precision_score(true_labels_np, predicted_labels_gram2vec, average=None, zero_division=0)
        gram2vec_recall_per_class = recall_score(true_labels_np, predicted_labels_gram2vec, average=None, zero_division=0)
        
        f1_per_class = f1_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)
        precision_per_class = precision_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)

        # Check if the directory exists, create it if not
        results_dir = "attention_residual_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        csv_file_path = f"attention_residual_results/results_{args.model_type}_{args.dataset}_{args.run_id}.csv"

        # Define the header for the CSV file
        csv_header = [
            "threshold", "g2v_correct/total", "residual_correct/total", "g2v_accuracy", "residual_accuracy",
            "gram2vec_diff_author_precision", "gram2vec_same_author_precision",
            "gram2vec_diff_author_recall", "gram2vec_same_author_recall",
            "gram2vec_diff_author_f1", "gram2vec_same_author_f1",
            "residual_diff_author_precision", "residual_same_author_precision",
            "residual_diff_author_recall", "residual_same_author_recall",
            "residual_diff_author_f1", "residual_same_author_f1"
        ]

        # Check if the CSV file already exists
        file_exists = os.path.isfile(csv_file_path)

        # Open the CSV file in append mode
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write the header only if the file does not exist
            if not file_exists:
                writer.writerow(csv_header)

            # Write the stats line to the CSV file
            writer.writerow([
                threshold, f"{g2v_correct}/{total}", f"{long_correct}/{total}",
                f"{g2v_correct/len(true_labels):.4f}", f"{long_correct/len(true_labels):.4f}",
                f"{gram2vec_precision_per_class[0]:.4f}", f"{gram2vec_precision_per_class[1]:.4f}",
                f"{gram2vec_recall_per_class[0]:.4f}", f"{gram2vec_recall_per_class[1]:.4f}",
                f"{gram2vec_f1_per_class[0]:.4f}", f"{gram2vec_f1_per_class[1]:.4f}",
                f"{precision_per_class[0]:.4f}", f"{precision_per_class[1]:.4f}",
                f"{recall_per_class[0]:.4f}", f"{recall_per_class[1]:.4f}",
                f"{f1_per_class[0]:.4f}", f"{f1_per_class[1]:.4f}"
            ])
    ### END EVALAUTION CODE ###

    # Calculate the ROC curve and AUC for residual_cosims
    fpr_residual, tpr_residual, _ = roc_curve(true_labels_np, residual_cosims)
    auc_residual = auc(fpr_residual, tpr_residual)

    # Calculate the ROC curve and AUC for gram2vec_cosims
    fpr_gram2vec, tpr_gram2vec, _ = roc_curve(true_labels_np, gram2vec_cosims)
    auc_gram2vec = auc(fpr_gram2vec, tpr_gram2vec)

    fpr_neural, tpr_neural, _ = roc_curve(true_labels_np, neural_cosims)
    auc_neural = auc(fpr_neural, tpr_neural)

    # print(f"gram2vec AUC: {auc_gram2vec:.3f}, residual AUC: {auc_residual:.3f}")
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr_residual, tpr_residual, color='darkorange', lw=2, label=f'Residual AUC = {auc_residual:.3f}')
    plt.plot(fpr_gram2vec, tpr_gram2vec, color='blue', lw=2, label=f'Gram2Vec AUC = {auc_gram2vec:.3f}')
    plt.plot(fpr_neural, tpr_neural, color='green', lw=2, label=f'Neural AUC = {auc_neural:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC Curve for {args.model_type} on {args.dataset}')
    plt.legend(loc="lower right")
    plt.grid()
    # Check if the directory exists, if not create it
    save_dir = os.path.dirname("attention_residual_results/graphs/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"attention_residual_results/graphs/{args.model_type}_{args.dataset}_{args.run_id}_residual_auc.png")
    # # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file_path)

    # # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(csv_header)

        # Write the stats line to the CSV file
        writer.writerow([
            threshold, f"{g2v_correct}/{total}", f"{long_correct}/{total}", 
            f"{gram2vec_f1_per_class[1]:.4f}", f"{f1_per_class[1]:.4f}",
            f"{g2v_correct/len(true_labels):.4f}", f"{long_correct/len(true_labels):.4f}",
            f"{precision_per_class[1]:.4f}", f"{recall_per_class[1]:.4f}",
            f"{precision_per_class[0]:.4f}", f"{recall_per_class[0]:.4f}", f"{f1_per_class[0]:.4f}",
            f"{gram2vec_precision_per_class[1]:.4f}", f"{gram2vec_recall_per_class[1]:.4f}",
            f"{gram2vec_precision_per_class[0]:.4f}", f"{gram2vec_recall_per_class[0]:.4f}",
            f"{gram2vec_f1_per_class[0]:.4f}"
        ])
    # Plotting the residuals
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_labels, bins=50, color='skyblue', edgecolor='black')
    plt.title('predicted residuals in test')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    plt.savefig(f"attention_residual_results/graphs/predicted_labels_{args.model_type}_{args.run_id}_{args.dataset}.png")