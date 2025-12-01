import torch
from torch.utils.data import DataLoader
from AttentionResidualModel import AttentionResidualDataset, AttentionResidualModel
import pandas as pd
from torch.optim import AdamW
from transformers import AutoModel
from tqdm.auto import tqdm
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import argparse
import os  
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time 


def cache_and_process_documents(train_df, dev_df, test_df, g2v_vectorizer, normalized=False):

    start_time = time.time()
    def gather_documents(df, g2v_vectorizer):
        documents = []
        uids = []

        # Collect all documents and UIDs first
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Collecting Documents"):
            try:
                if pd.isna(row['document1']) or row['document1'] is None:
                    print(f"Warning: Found null document at index {i}")
                    continue
            except Exception as e:
                print(f"Error processing document at index {i}: {str(e)}")
                print(f"Document content: {repr(row['document1'])}")
                raise

            try:
                if pd.isna(row['document2']) or row['document2'] is None:
                    print(f"Warning: Found null document at index {i}")
                    continue
            except Exception as e:
                print(f"Error processing document at index {i}: {str(e)}")
                print(f"Document content: {repr(row['document2'])}")
                raise
            
            documents.extend([row['document1'], row['document2']])
            uids.extend([row['documentID1'], row['documentID2']])
        
        return documents, uids
    
    def cache_documents(documents, uids, g2v_vectorizer):
        # Process documents in batches
        batch_size = 200
        n_batches = len(documents) // batch_size + (1 if len(documents) % batch_size else 0)
        
        for i in tqdm(range(n_batches), desc="Vectorizing and Caching documents", total=n_batches):
            if i % 200 == 0:
                print(f"Processing batch {i+1} of {n_batches}")
                g2v_vectorizer.save_cache()
            
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(documents))
            
            doc_batch = documents[start_idx:end_idx]
            uid_batch = uids[start_idx:end_idx]
            
            g2v_vectorizer.cache_vectors_batch(doc_batch, uid_batch)
        
        g2v_vectorizer.save_cache()

    # Process each dataset
    train_documents, train_uids = gather_documents(train_df, g2v_vectorizer)
    dev_documents, dev_uids = gather_documents(dev_df, g2v_vectorizer)
    test_documents, test_uids = gather_documents(test_df, g2v_vectorizer)
    combined_documents = train_documents + dev_documents + test_documents
    combined_uids = train_uids + dev_uids + test_uids
    cache_documents(combined_documents, combined_uids, g2v_vectorizer)
    g2v_vectorizer.save_cache()

    ### returns: data, a list of dictionaries with the relevant data: text1, text2, features1, features2, residual
    def process_posts(df, g2v_vectorizer, normalized=False):
        data = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Posts"):
            features1, features2, cosim = g2v_vectorizer.get_vector_and_score(row['document1'], row['document2'], row['documentID1'], row['documentID2'], normalized=normalized)
            gold = row['same_author_label']
            if gold == 1 or gold == True:
                residual = (1 - cosim)
                # print(gold)   
            elif gold == 0 or gold == False or gold == -1:
                residual = (-1 - cosim)
            data.append({"text1":row['document1'],
                            "text2":row['document2'],
                            "features1":features1,
                            "features2":features2,
                            "residual":residual,
                            "same":row['same_author_label']})
        
        return data
    
    train_data = process_posts(train_df, g2v_vectorizer, normalized=normalized)
    dev_data = process_posts(dev_df, g2v_vectorizer, normalized=normalized)
    test_data = process_posts(test_df, g2v_vectorizer, normalized=normalized)

    print(f"Time to process documents: {time.time() - start_time:.2f} seconds")
    return train_data, dev_data, test_data

def generate_neural_feature_map(dataloader, model_type):
    if model_type == "/home/pezeng/rsp/LUAR-RU":
        model = AutoModel.from_pretrained(model_type, local_files_only=True, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_type, trust_remote_code=True)

    model.to(device)
    neural_cosims = []
    model.eval()
    with torch.no_grad():
        dataloader = tqdm(dataloader, desc="test loop", position=1 ,leave=True)

        for batch in dataloader:
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
    parser.add_argument("-m", "--model_type", type=str, default="luar", choices=["roberta", "roberta-large", "luar", "style", "luar-ru", "ruRoPEBert", "all-mpnet-base-v2", "mxbai-embed-large-v1", "longformer"], help="Type of model to use for training.")
    parser.add_argument("-d", "--dataset", type=str, default="reddit", choices=["combined", "fanfiction", "reddit", "amazon", "hiatus", "imbalanced_reddit","hiatus_combined","hiatus_russian","pikabu","new_hiatus_combined", "reddit-amazon-fanfic", "biber"], help="Dataset to use for training.")
    parser.add_argument("-r", "--run_id", type=str, required=False, help="Run ID for the experiment.")
    parser.add_argument("-p", "--percentage", type=float, default=1, help="Percentage of data to sample for training, validation, and testing.")
    parser.add_argument("-c", "--config", type=str, default="config.txt", help="Path to the config file.")
    parser.add_argument("-s", "--save_dir", type=str, default="vector_cache", help="Path to the save directory.")
    parser.add_argument("-f", "--feature_dim", type=int, default=618, help="Feature dimension.")
    parser.add_argument("-n", "--normalized", action="store_true", default=True, help="Whether to use normalized vectors.")
    parser.add_argument("-ln", "--layernorm", type=str, required=False, choices=["pre", "post", "both"], help="Whether to use pre or post layernorm.")
    parser.add_argument("-l", "--language", type=str, default="en", choices=["en", "ru"], help="Language of the dataset.")
    parser.add_argument("-k", "--fold", type=int, required=False, help="Fold number for k-fold cross validation")
    parser.add_argument("-v", "--vector_cache_fp", type=str, required=False, help="Path to the vector cache file.")
    args = parser.parse_args()

    print(f"normalized: {args.normalized}")
    if not args.run_id:
        date = datetime.now().strftime("%Y-%m-%d")
        args.run_id = date
    
    n = "normalized" if args.normalized else "not_normalized"
    ln = args.layernorm if args.layernorm else "no_ln"
    # Create descriptive experiment name
    settings_folder = f"{args.model_type}_{args.dataset}_{n}_{ln}"

    if args.fold is not None:
        experiment_name = f"{args.run_id}_fold{args.fold}"
        data_base_path = f"../data/{args.dataset}_kfold/fold_{args.fold}"
    else:
        experiment_name = f"{args.run_id}"
        data_base_path = f"../data/{args.dataset}"

    print("settings_folder: ", settings_folder)
    print("experiment_name: ", experiment_name)
    # Create output directories
    base_output_dir = f"../experiments/{settings_folder}/{experiment_name}"
    # slurm_output_dir = f"../slurm_outputs/{settings_folder}/{experiment_name}"
    model_dir = f"{base_output_dir}/models"
    results_dir = f"{base_output_dir}/results"
    graph_dir = f"{base_output_dir}/graphs"
    
    # os.makedirs(slurm_output_dir, exist_ok=True)

    # Create all directories
    for directory in [model_dir, results_dir, graph_dir]:
        os.makedirs(directory, exist_ok=True)

    # exit()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Update file paths to use fold-specific data

    train_df = pd.read_csv(f"{data_base_path}/train.csv", encoding="utf-8")
    dev_df = pd.read_csv(f"{data_base_path}/dev.csv", encoding="utf-8")
    test_df = pd.read_csv(f"{data_base_path}/test.csv", encoding="utf-8")
    
    if args.language == "en":
        os.environ["LANGUAGE"] = "en"
        os.environ["SPACY_MODEL"] = "en_core_web_lg"
    elif args.language == "ru":
        os.environ["LANGUAGE"] = "ru"
        os.environ["SPACY_MODEL"] = "ru_core_news_lg"
    
    from explainable_module import Gram2VecModule
    
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
        "num_tokens":1
    }

    ### CACHE IS DECIDED HERE ###
    if args.vector_cache_fp is not None:
        g2v_vectorizer = Gram2VecModule(filepath=args.vector_cache_fp, dataset=args.dataset, save_dir=args.save_dir, run_id=args.run_id, configs=vectorizer_configs)
    else:
        if args.fold is not None:
            g2v_vectorizer = Gram2VecModule(filepath=f"vector_cache/{args.dataset}_{args.run_id}_fold{args.fold}_vector_map.pkl", dataset=args.dataset, save_dir=args.save_dir, run_id=args.run_id, configs=vectorizer_configs)
        else:
            g2v_vectorizer = Gram2VecModule(filepath=f"vector_cache/{args.dataset}_{args.run_id}_vector_map.pkl", dataset=args.dataset, save_dir=args.save_dir, run_id=args.run_id, configs=vectorizer_configs)
    # if not os.path.exists(f"vector_cache/{args.dataset}_{args.run_id}_normalized_vector_map.pkl"):
        # g2v_vectorizer.normalize_cache()
        # g2v_vectorizer.save_cache(normalized=True)

    if args.model_type == "luar":
        model_type = "rrivera1849/LUAR-MUD"
    elif args.model_type == "luar-ru":
        model_type = "/home/pezeng/rsp/LUAR-RU"
    elif args.model_type == "roberta":
        model_type = "FacebookAI/roberta-base"
    elif args.model_type == "roberta-large":
        model_type = "FacebookAI/roberta-large"
    elif args.model_type == "style":
        model_type = "AnnaWegmann/Style-Embedding"
    elif args.model_type == "longformer":
        model_type = "allenai/longformer-base-4096"
    elif args.model_type == "ruRoPEBert":
        model_type = "Tochka-AI/ruRoPEBert-e5-base-2k"
    elif args.model_type == "all-mpnet-base-v2":
        model_type = "sentence-transformers/all-mpnet-base-v2"
    elif args.model_type == "mxbai-embed-large-v1":
        model_type = "mixedbread-ai/mxbai-embed-large-v1"
        
    train_df_sampled = train_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility
    dev_df_sampled = dev_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility
    test_df_sampled = test_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility

    train_data, dev_data, test_data = cache_and_process_documents(train_df_sampled, dev_df_sampled, test_df_sampled, g2v_vectorizer, normalized=args.normalized)
    
    model = AttentionResidualModel(model_type=model_type, feature_dim=args.feature_dim)

    # Load the model from the checkpoint
    # model.load_state_dict(torch.load(f"/home/pezeng1/dev/rsp/experiments/luar_new_hiatus_combined_normalized_no_ln/fix_norm/models/checkpoint.pt")['model_state_dict'])

    model.to(device)
    print(model_type)
    train_dataset = AttentionResidualDataset(train_data, model_type = model_type)
    dev_dataset = AttentionResidualDataset(dev_data, model_type = model_type)
    test_dataset = AttentionResidualDataset(test_data, model_type = model_type)

    # Optimized for A6000 GPU 
    if args.model_type == "roberta-large" or args.model_type == "longformer" or args.model_type == "mxbai-embed-large-v1":
        b_size = 32
    elif args.model_type == "style" or args.model_type == "all-mpnet-base-v2":
        b_size = 64
    elif args.model_type == 'luar' or args.model_type == "roberta": 
        b_size = 128
    elif args.model_type == "luar-ru":
        b_size = 128
    elif args.model_type == "ruRoPEBert":
        b_size = 128
    else:
        b_size = 64
    
    accumulation_steps = 1
    print(f"Using batch size of {b_size}")
    print(f"number of accuulation steps: {accumulation_steps}")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=b_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=b_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    if args.model_type == "luar-ru":
        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=.0001)  # Lower learning rate for Russian model
    else:
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=.0001)  # Default learning rate
    # optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=.0001)  # Slightly higher initial learning rate
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Add this before the training loop
    attention_log = []  # Store attention weights for all epochs

    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 3

    # Define the total number of epochs
    num_epochs = 15
    
    # Track training and validation losses
    train_losses = []
    val_losses = []
    
    pbar = tqdm(total=num_epochs, desc="Overall Training Progress", position=0)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_dataloader = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", position=1, leave=True)
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            doc1 = batch['text1']
            doc2 = batch['text2']
            features1 = batch['features1']
            features2 = batch['features2']
            labels = batch["labels"].to(device)
            outputs = model(doc1, doc2, features1, features2, labels=labels, layernorm="pre")
            loss = outputs["loss"] / accumulation_steps  # Normalize loss to account for accumulation
            
            # Use regular backward
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                # Remove scaler, use regular optimizer step
                optimizer.step()
                optimizer.zero_grad()

            # Account for batch size when accumulating loss
            total_loss += (loss.detach().item() * accumulation_steps * len(labels))
            train_dataloader.set_postfix(loss=(total_loss / ((i + 1) * len(labels))), refresh=False)

        avg_loss = total_loss / len(train_dataset)
        train_losses.append(avg_loss)  # Store training loss
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
                doc1 = batch['text1']
                doc2 = batch['text2']
                features1 = batch['features1']
                features2 = batch['features2']
                labels = batch["labels"].to(device)
                outputs = model(doc1, doc2, features1, features2, labels=labels, layernorm="pre")
                loss = outputs["loss"]
                
                # Multiply by batch size to get total loss for the batch
                val_loss += loss.item() * len(labels)
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

        # Divide by total number of samples instead of number of batches
        avg_val_loss = val_loss / len(dev_dataset)
        val_losses.append(avg_val_loss)  # Store validation loss
        print(f"Validation Loss: {avg_val_loss}")

        # In the training loop where we handle attention weights
        if attention_weights_count > 0:
            avg_attention_weights = attention_weights_sum / attention_weights_count
            print("\nAverage Attention Weights for hidden1, hidden2, features1, features2:")
            print(avg_attention_weights)
            
            # Store the attention weights for this epoch
            epoch_log = f"\nEpoch {epoch+1} Attention Weights:\n"
            epoch_log += "-" * 50 + "\n"
            input_names = ['hidden1', 'hidden2', 'features1', 'features2']
            for i, name in enumerate(input_names):
                epoch_log += f"Attention weight for {name}: {avg_attention_weights[i]:.4f}\n"
            epoch_log += f"\nValidation Loss: {avg_val_loss:.4f}\n"
            epoch_log += "-" * 50 + "\n"
            attention_log.append(epoch_log)

        logging.info(f'Epoch {epoch+1}/{num_epochs} completed.')
        # Save the model if the validation loss is the best we've seen so far.
        if avg_val_loss < best_val_loss:
            print("saving model")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_dir}/model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, f"{model_dir}/checkpoint.pt")
            early_stopping_counter = 0  # reset counter after improvement
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        # scheduler.step(avg_val_loss)

        pbar.update(1)

    pbar.close()

    # After the training loop (before testing), add:
    attention_file = f"{results_dir}/attention_weights.txt"
    with open(attention_file, 'w') as f:
        f.write("Training Attention Weights Log\n")
        f.write("=" * 50 + "\n")
        f.writelines(attention_log)

    # Save hyperparameters and training configuration
    config_file = f"{results_dir}/training_config.txt"
    with open(config_file, 'w') as f:
        f.write("TRAINING CONFIGURATION\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch Size: {b_size}\n")
        f.write(f"Accumulation Steps: {accumulation_steps}\n")
        
        # Determine if we're finetuning
        is_finetuning = args.dataset != "hiatus_combined"
        f.write(f"\nMode: {'FINE-TUNING' if is_finetuning else 'INITIAL TRAINING'}\n")
        if is_finetuning:
            f.write("Loading weights from hiatus_combined checkpoint\n")
            f.write(f"Learning Rate: 1e-5 (reduced for finetuning)\n")
            f.write(f"Number of Epochs: 10 (reduced for finetuning)\n")
        
        f.write(f"Feature Dimension: {args.feature_dim}\n")
        f.write(f"Normalized: {args.normalized}\n")
        f.write(f"LayerNorm: {args.layernorm if args.layernorm else 'none'}\n")
        f.write("=" * 50 + "\n")

    # neural_cosims = generate_neural_feature_map(test_dataloader, model_type)

    # TESTING LOOP TO SEE HOW MUCH FINETUNED MODEL CORRECTS GRAM2VEC:
    gram2vec_cosims = []
    for item in test_data:
        features1 = item['features1']
        features2 = item['features2']
        cosim = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
        gram2vec_cosims.append(cosim[0][0])

    predicted_labels = [] 

    model.eval()
    with torch.no_grad():
        test_dataloader = tqdm(test_dataloader, desc="test loop", position=1 ,leave=True)

        for batch in test_dataloader:
            # print(batch['labels'])
            doc1 = batch['text1']
            doc2 = batch['text2']
            features1 = batch['features1']
            features2 = batch['features2']
            labels = batch["labels"].to(device)  # Move labels to the device
            outputs = model(doc1, doc2, features1, features2, labels=labels, layernorm="pre")
            # print(outputs)
            # print(outputs["logits"])
            predictions = outputs["logits"].squeeze().detach().cpu().numpy()  # Adjust based on your model's output
            predicted_labels.append(predictions)

    residual_cosims = [gram2vec_cosims[i] + predicted_labels[i] for i in range(len(predicted_labels))]

    ic = [1 - abs(predicted_labels[i]) for i in range(len(predicted_labels))]
    # Save predictions and related data
    predictions_df = pd.DataFrame({
        'document1': test_df_sampled['document1'],
        'document2': test_df_sampled['document2'],
        'documentID1': test_df_sampled['documentID1'],
        'documentID2': test_df_sampled['documentID2'],
        'true_label': test_df_sampled['same_author_label'],
        'gram2vec_score': gram2vec_cosims,
        # 'neural_score': neural_cosims,
        'predicted_residual': predicted_labels,
        'final_score': residual_cosims,
        'ic': ic
    })
    
    # Save to CSV in the results directory
    predictions_file = f"{results_dir}/predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Saved predictions to {predictions_file}")
    ### EVALAUTION CODE ###
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for threshold in thresholds:
        g2v_correct = 0
        long_correct = 0
        true_labels = list(test_df_sampled['same_author_label'])

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

        csv_file_path = f"{results_dir}/metrics.csv"

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

    # fpr_neural, tpr_neural, _ = roc_curve(true_labels_np, neural_cosims)
    # auc_neural = auc(fpr_neural, tpr_neural)

    # print(f"gram2vec AUC: {auc_gram2vec:.3f}, residual AUC: {auc_residual:.3f}")
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr_residual, tpr_residual, color='darkorange', lw=2, label=f'Residual AUC = {auc_residual:.3f}')
    plt.plot(fpr_gram2vec, tpr_gram2vec, color='blue', lw=2, label=f'Gram2Vec AUC = {auc_gram2vec:.3f}')
    # plt.plot(fpr_neural, tpr_neural, color='green', lw=2, label=f'Neural AUC = {auc_neural:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC Curve for {args.model_type} on {args.dataset}')
    plt.legend(loc="lower right")
    plt.grid()

    plt.savefig(f"{graph_dir}/auc_curve.png")

    # Plotting the residuals
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_labels, bins=50, color='skyblue', edgecolor='black')
    plt.title('predicted residuals in test')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    plt.savefig(f"{graph_dir}/predicted_labels.png")

    # After training loop, plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{graph_dir}/loss_curves.png")
    print(f"Saved loss curves to {graph_dir}/loss_curves.png")