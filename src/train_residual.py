'''
    This code trains a model to predict the residual 
    between ground truth and predicted cosine similarities from interpretable system.
'''
import torch
from torch.utils.data import DataLoader
import pandas as pd
from torch.optim import AdamW
from tqdm.auto import tqdm
from ResidualModel import ResidualDataset, ResidualModel
from explainable_module import Gram2VecModule
from torch.cuda.amp import autocast
import logging
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import numpy as np
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
import torch
import os
import csv
from sklearn.metrics import roc_curve, auc, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
    parser.add_argument("-m", "--model_type", type=str, default="roberta", choices=["longformer", "roberta", "luar", "roberta-large", "style"], help="Type of model to use for training.")
    parser.add_argument("-d", "--dataset", type=str, default="reddit", choices=["fanfiction", "reddit", "amazon"], help="Dataset to use for training.")
    parser.add_argument("-r", "--run_id", type=str, default="0", help="Run ID for the experiment.")
    args = parser.parse_args()

    g2v_vectorizer = Gram2VecModule(dataset=args.dataset)

    model = ResidualModel(model_type=args.model_type)
    print(model.enc_model.print_trainable_parameters())

    model.to(device)

    train_df = pd.read_csv(f"../data/{args.dataset}/train.csv", encoding="utf-8", lineterminator='\n')
    dev_df = pd.read_csv(f"../data/{args.dataset}/dev.csv", encoding="utf-8", lineterminator='\n')
    test_df = pd.read_csv(f"../data/{args.dataset}/test.csv", encoding="utf-8", lineterminator='\n')

    train_data = process_posts(train_df, g2v_vectorizer)
    dev_data = process_posts(dev_df, g2v_vectorizer)
    test_data = process_posts(test_df, g2v_vectorizer)

    if not os.path.exists(f'{args.dataset}_vector_map.pkl'):  
        g2v_vectorizer.save_cache()

    train_dataset = ResidualDataset(train_data, model_type = args.model_type)
    dev_dataset = ResidualDataset(dev_data, model_type = args.model_type)
    test_dataset = ResidualDataset(test_data, model_type = args.model_type)
    
    # Optimized for A6000 GPU
    if args.model_type == 'roberta-large' or args.model_type == 'longformer':
        b_size = 32
    elif args.model_type == 'roberta' or args.model_type == 'style':
        b_size = 64
    elif args.model_type == 'luar':
        b_size = 128

    accumulation_steps = 1
    print(f"Using batch size of {b_size}")
    print(f"number of accuulation steps: {accumulation_steps}")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=b_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=b_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=.0001)

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
                context = {k: v.to(device) for k, v in batch["context"].items()}
                labels = batch["labels"].to(device)  
                outputs = model(context, labels=labels)
                loss = outputs["loss"] / accumulation_steps  
            
            # print(outputs['logits'])
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()  
                optimizer.zero_grad()  

            total_loss += loss.item() * accumulation_steps  
            train_dataloader.set_postfix(loss=(total_loss / (i + 1)), refresh=False)

        avg_loss = total_loss/len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")

        model.eval()
        val_loss = 0

        with torch.no_grad():
            dev_dataloader = tqdm(dev_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", position=1 ,leave=True)

            for batch in dev_dataloader:
                with autocast():
                    context = {k: v.to(device) for k, v in batch["context"].items()}
                    labels = batch["labels"].to(device)  
                    outputs = model(context, labels=labels)
                    loss = outputs["loss"]
                
                val_loss += loss.item()

        avg_val_loss = val_loss / len(dev_dataloader)
        print(f"Validation Loss: {avg_val_loss}")

        logging.info(f'Epoch {epoch+1}/{num_epochs} completed.')
        # Save the model if the validation loss is the best we've seen so far.
        if avg_val_loss < best_val_loss:
            print("saving model")
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                }, f"../model/{args.model_type}_{args.dataset}_{args.run_id}_residual_checkpoint.pt")
            early_stopping_counter = 0  
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        pbar.update(1)

    pbar.close()

    ### THRESHOLD SELECTION ###
    print("Selecting Threshold")
    gram2vec_cosims = []
    vector_map = pd.read_pickle(f'{args.dataset}_vector_map.pkl')

    for i, row in train_df.iterrows():
        vector1 = vector_map[row['post_1_id']]
        vector2 = vector_map[row['post_2_id']]
        cosim = cosine_similarity(vector1, vector2)
        gram2vec_cosims.append(cosim[0][0])

    predicted_labels = []
    print("length of gram2vec_cosims is: ", len(gram2vec_cosims))
    model.eval()
    with torch.no_grad():
        train_dataloader = tqdm(train_dataloader, desc="test loop", position=1, leave=True)

        for batch in train_dataloader:
            
            with autocast():
                context = {k: v.to(device) for k, v in batch["context"].items()}
                labels = batch["labels"].to(device)  
                outputs = model(context, labels=labels)
        
            predictions = outputs["logits"].squeeze().detach().cpu().numpy() 
            predicted_labels.extend(predictions)

    residual_cosims = [gram2vec_cosims[i] + predicted_labels[i] for i in range(len(predicted_labels))]
    true_labels_np = np.array(list(train_df['same']))

    thresholds = np.linspace(0.1, 1, 19)
    best_threshold = 0
    best_f1 = 0

    for threshold in thresholds:
        print(threshold)
        predicted_labels_residual = np.array([1 if score > threshold else 0 for score in residual_cosims])
        f1_per_class = f1_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)
        
        # Check if the current threshold gives a better same author F1 score
        if f1_per_class[1] > best_f1:
            best_f1 = f1_per_class[1]
            best_threshold = threshold

    # Print data types of variables
    print(f"Data type of best_threshold: {type(best_threshold)}")
    print(f"Data type of best_f1: {type(best_f1)}")
    print(f"Data type of f1_per_class: {type(f1_per_class)}")
    print(f"Best threshold for {args.model_type} on {args.dataset}: {best_threshold}")
    ### END THRESHOLD SELECTION ###
    
    ### Calculate residuals for test set
    gram2vec_cosims = []
    vector_map = pd.read_pickle(f'{args.dataset}_vector_map.pkl')

    for i, row in test_df.iterrows():
        vector1 = vector_map[row['post_1_id']]
        vector2 = vector_map[row['post_2_id']]
        cosim = cosine_similarity(vector1, vector2)
        gram2vec_cosims.append(cosim[0][0])

    predicted_labels = []

    model.eval()
    with torch.no_grad():
        test_dataloader = tqdm(test_dataloader, desc="test loop", position=1 ,leave=True)

        for batch in test_dataloader:
            
            with autocast():
                context = {k: v.to(device) for k, v in batch["context"].items()}
                labels = batch["labels"].to(device)  
                outputs = model(context, labels=labels)
         
            predictions = outputs["logits"].squeeze().detach().cpu().numpy() 
            predicted_labels.append(predictions)

    residual_cosims = [gram2vec_cosims[i] + predicted_labels[i] for i in range(len(predicted_labels))]
    same_labels = list(test_df['same'])
    
    ### ANALYSIS CODE ###
    test_data_for_csv = []
    # Collect data for CSV
    for i in range(len(predicted_labels)):
        test_data_for_csv.append([gram2vec_cosims[i], predicted_labels[i], residual_cosims[i], same_labels[i]])

    # Define the CSV file path
    csv_file_path = f"../results/residual_predictions_{args.model_type}_{args.run_id}_{args.dataset}.csv"

    # Define the header for the CSV file
    csv_header = ["gram2vec cosim", "predicted residual", "corrected cosim", 'same']

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        for row in test_data_for_csv:
            writer.writerow(row)  # Add None for the threshold column

    print(f"Prediction results on Test set for analysis saved to {csv_file_path}")
    ### END ANALYSIS CODE ###

    ### EVALAUTION CODE ###
    g2v_correct = 0
    residual_correct = 0
    true_labels = list(test_df['same'])
    total = len(true_labels)

    for i in range(total):
        if true_labels[i] == 1:
            if gram2vec_cosims[i] > best_threshold:
                g2v_correct += 1
            if residual_cosims[i] > best_threshold:
                residual_correct += 1
        elif true_labels[i] == 0:
            if gram2vec_cosims[i] < best_threshold:
                g2v_correct += 1
            if residual_cosims[i] < best_threshold:
                residual_correct += 1

    # Convert lists to numpy arrays for sklearn metrics
    true_labels_np = np.array(true_labels)
    predicted_labels_residual = np.array([1 if score > best_threshold else 0 for score in residual_cosims])
    predicted_labels_gram2vec = np.array([1 if score > best_threshold else 0 for score in gram2vec_cosims])

    # Calculate F1 scores
    gram2vec_f1_per_class = f1_score(true_labels_np, predicted_labels_gram2vec, average=None, zero_division=0)
    residual_f1_per_class = f1_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)

    # Calculate accuracies
    gram2vec_accuracy = g2v_correct / total
    residual_accuracy = residual_correct / total

    # Print statistics in the specified order
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Gram2Vec same author F1: {gram2vec_f1_per_class[1]:.4f}")
    print(f"Residual same author F1: {residual_f1_per_class[1]:.4f}")
    print(f"Gram2Vec different author F1: {gram2vec_f1_per_class[0]:.4f}")
    print(f"Residual different author F1: {residual_f1_per_class[0]:.4f}")
    print(f"Gram2Vec correct: {g2v_correct}/{total}")
    print(f"Residual correct: {residual_correct}/{total}")
    print(f"Gram2Vec accuracy: {gram2vec_accuracy:.4f}")
    print(f"Residual accuracy: {residual_accuracy:.4f}")

    # Calculate the ROC curve and AUC for residual_cosims
    fpr_residual, tpr_residual, _ = roc_curve(true_labels_np, residual_cosims)
    auc_residual = auc(fpr_residual, tpr_residual)

    # Calculate the ROC curve and AUC for gram2vec_cosims
    fpr_gram2vec, tpr_gram2vec, _ = roc_curve(true_labels_np, gram2vec_cosims)
    auc_gram2vec = auc(fpr_gram2vec, tpr_gram2vec)

    print(f"gram2vec AUC: {auc_gram2vec:.3f}, residual AUC: {auc_residual:.3f}")
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr_residual, tpr_residual, color='darkorange', lw=2, label=f'Residual AUC = {auc_residual:.3f}')
    plt.plot(fpr_gram2vec, tpr_gram2vec, color='blue', lw=2, label=f'Gram2Vec AUC = {auc_gram2vec:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC Curve for {args.model_type} on {args.dataset}')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"../results/residual/graphs/residual/{args.model_type}_{args.dataset}_{args.run_id}_residual_auc.png")