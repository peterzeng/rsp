import torch
from torch.utils.data import DataLoader
from AttentionResidualModel import AttentionResidualDataset, AttentionResidualModel
import pandas as pd
from transformers import AutoModel
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torch.amp import autocast
from sklearn.metrics import roc_curve, auc
import argparse
import os  
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def cache_and_process_documents(test_df, g2v_vectorizer, normalized=False):

    def cache_documents(df, g2v_vectorizer):
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Updating Vector Cache"):
            try:
                if pd.isna(row['document1']) or row['document1'] is None:
                    print(f"Warning: Found null document at index {i}")
                    continue
            except Exception as e:
                print(f"Error processing document at index {i}: {str(e)}")
                print(f"Document content: {repr(row['document1'])}")
                raise  # Re-raise the exception if you want to stop execution

            try:
                if pd.isna(row['document2']) or row['document2'] is None:
                    print(f"Warning: Found null document at index {i}")
                    continue
                    
            except Exception as e:
                print(f"Error processing document at index {i}: {str(e)}")
                print(f"Document content: {repr(row['document2'])}")
                raise  # Re-raise the exception if you want to stop execution
                
            g2v_vectorizer.cache_vector(row['document1'], row['document1_id'])
            g2v_vectorizer.cache_vector(row['document2'], row['document2_id'])

    cache_documents(test_df, g2v_vectorizer)
    
    ### returns: data, a list of dictionaries with the relevant data: text1, text2, features1, features2, residual
    def process_posts(df, g2v_vectorizer, normalized=False):
        data = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Posts"):
            features1, features2, cosim = g2v_vectorizer.get_vector_and_score(row['document1'], row['document2'], row['document1_id'], row['document2_id'], normalized=normalized)
            gold = row['same_author_label']
            if gold == 1 or gold == True:
                residual = (1 - cosim)
            elif gold == 0 or gold == False:
                residual = (-1 - cosim)
            data.append({"text1":row['document1'],
                            "text2":row['document2'],
                            "features1":features1,
                            "features2":features2,
                            "residual":residual,
                            "same":row['same_author_label']})
        
        return data
    
    test_data = process_posts(test_df, g2v_vectorizer, normalized=normalized)

    return test_data

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
            with autocast(device_type='cuda'):
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
    parser.add_argument("-mfp", "--model_fp", type=str, required=False, help="Path to the model file.")
    parser.add_argument("-d", "--dataset", type=str, default="reddit", choices=["fanfiction", "reddit", "amazon", "hiatus", "hiatus_combined","hiatus_russian","pikabu"], help="Dataset to use for training.")
    parser.add_argument("-r", "--run_id", type=str, required=False, help="Run ID for the experiment.")
    parser.add_argument("-s", "--save_dir", type=str, default="vector_cache", help="Path to the save directory.")
    parser.add_argument("-f", "--feature_dim", type=int, default=617, help="Feature dimension.")
    parser.add_argument("-n", "--normalized", action="store_true", default=True,help="Whether to use normalized vectors.")
    parser.add_argument("--layernorm", type=str, required=False, choices=["pre", "post", "both"], help="Whether to use pre or post layernorm.")
    parser.add_argument("-l", "--language", type=str, default="en", choices=["en", "ru"], help="Language of the dataset.")
    parser.add_argument("-k", "--fold", type=int, required=False, help="Fold number for k-fold cross validation")
    parser.add_argument("-v", "--vector_cache_fp", type=str, required=False, help="Path to the vector cache file.")
    args = parser.parse_args()

    if not args.run_id:
        date = datetime.now().strftime("%Y-%m-%d")
        args.run_id = date
    
    n = "normalized" if args.normalized else "not_normalized"
    ln = args.layernorm if args.layernorm else "no_ln"
    print(f"Running {args.model_type} on {args.dataset} with {n} and {ln}")

    # Create descriptive experiment name
    settings_folder = f"{args.model_type}_{args.dataset}_{n}_{ln}"
    if args.fold is not None:
        experiment_name = f"{args.run_id}_fold{args.fold}"
        data_base_path = f"../data/{args.dataset}_kfold/fold_{args.fold}"
    else:
        experiment_name = f"{args.run_id}"
        data_base_path = f"../data/{args.dataset}"

    print("experiment_name: ", experiment_name)
    # Create output directories
    base_output_dir = f"../experiments/{settings_folder}/{experiment_name}"
    model_dir = f"{base_output_dir}/models"
    results_dir = f"{base_output_dir}/results"
    graph_dir = f"{base_output_dir}/graphs"
    
    # slurm_output_dir = f"../slurm_outputs/{settings_folder}/{experiment_name}"
    # os.makedirs(slurm_output_dir, exist_ok=True)

    # Create all directories
    for directory in [model_dir, results_dir, graph_dir]:
        os.makedirs(directory, exist_ok=True)

    # exit()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Update file paths to use fold-specific data
    if args.fold is not None:
        data_base_path = f"../data/{args.dataset}_kfold/fold_{args.fold}"
    else:
        data_base_path = f"../data/{args.dataset}"

    print(f"data_base_path: {data_base_path}")
    test_df = pd.read_csv(f"{data_base_path}/test.csv", encoding="utf-8", lineterminator='\n')
    print(f"columns: {test_df.columns}")
   
    if args.language == "en":
        os.environ["LANGUAGE"] = "en"
        os.environ["SPACY_MODEL"] = "en_core_web_lg"
    elif args.language == "ru":
        os.environ["LANGUAGE"] = "ru"
        os.environ["SPACY_MODEL"] = "ru_core_news_lg"
    
    from explainable_module import Gram2VecModule
    
    ### CACHE IS DECIDED HERE ###
    if args.vector_cache_fp is not None:
        g2v_vectorizer = Gram2VecModule(filepath=args.vector_cache_fp, dataset=args.dataset, save_dir=args.save_dir, run_id=args.run_id, configs=None)
    else:
        if args.fold is not None:
            g2v_vectorizer = Gram2VecModule(filepath=f"vector_cache/{args.dataset}_{args.run_id}_fold{args.fold}_vector_map.pkl", dataset=args.dataset, save_dir=args.save_dir, run_id=args.run_id, configs=None)
        else:
            g2v_vectorizer = Gram2VecModule(filepath=f"vector_cache/{args.dataset}_{args.run_id}_vector_map.pkl", dataset=args.dataset, save_dir=args.save_dir, run_id=args.run_id, configs=None)

    if args.model_type == "luar":
        model_type = "rrivera1849/LUAR-MUD"
    elif args.model_type == "luar-ru":
        model_type = "/home/pezeng/rsp/LUAR-RU"
    elif args.model_type == "roberta":
        model_type = "sentence-transformers/all-distilroberta-v1"
    elif args.model_type == "roberta-large":
        model_type = "sentence-transformers/all-roberta-large-v1"
    elif args.model_type == "style":
        model_type = "AnnaWegmann/Style-Embedding"

    if args.model_fp is not None:
        model = AttentionResidualModel(model_type=model_type, feature_dim=args.feature_dim)
        model.load_state_dict(torch.load(args.model_fp))
    else:
        model = AttentionResidualModel(model_type=model_type, feature_dim=args.feature_dim)
        model.load_state_dict(torch.load(f"{model_dir}/model.pt"))
    model.to(device)
    
    test_data = cache_and_process_documents(test_df, g2v_vectorizer, normalized=args.normalized)

    print(model_type)
    test_dataset = AttentionResidualDataset(test_data, model_type = model_type)

    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # neural_cosims = generate_neural_feature_map(test_dataloader, model_type)

    # TESTING LOOP TO SEE HOW MUCH FINETUNED MODEL CORRECTS GRAM2VEC:
    gram2vec_cosims = []
    for item in test_data:
        features1 = item['features1']
        features2 = item['features2']
        cosim = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
        gram2vec_cosims.append(cosim[0][0])

    predicted_labels = []
    attention_weights_sum = None
    attention_weights_count = 0

    model.eval()
    with torch.no_grad():
        test_dataloader = tqdm(test_dataloader, desc="test loop", position=1 ,leave=True)

        for batch in test_dataloader:
            with autocast(device_type='cuda'):
                doc1 = batch['text1']
                doc2 = batch['text2']
                features1 = batch['features1']
                features2 = batch['features2']
                labels = batch["labels"].to(device)
                outputs = model(doc1, doc2, features1, features2, labels=labels, layernorm="pre")

            predictions = outputs["logits"].squeeze().detach().cpu().numpy()
            predicted_labels.append(predictions)

            # Accumulate attention weights
            batch_attention = outputs["attention_weights"].mean(dim=1)  # Average across heads
            if attention_weights_sum is None:
                attention_weights_sum = batch_attention.sum(dim=0).cpu().numpy()
            else:
                attention_weights_sum += batch_attention.sum(dim=0).cpu().numpy()
            attention_weights_count += batch_attention.shape[0]

    # Calculate and print average test attention weights
    if attention_weights_count > 0:
        avg_attention_weights = attention_weights_sum / attention_weights_count
        print("\nAverage Test Attention Weights:")
        print(avg_attention_weights)
        
        print("\nTest Attention Weight Interpretations:")
        input_names = ['hidden1', 'hidden2', 'features1', 'features2']
        for i, name in enumerate(input_names):
            print(f"Attention weight for {name}: {avg_attention_weights[i]:.4f}")

    residual_cosims = [gram2vec_cosims[i] + predicted_labels[i] for i in range(len(predicted_labels))]
    same_labels = list(test_df['same_author_label'])
    
    ic = [1 - abs(predicted_labels[i]) for i in range(len(predicted_labels))]
    
    # Save predictions and related data
    predictions_df = pd.DataFrame({
        'document1': test_df['document1'],
        'document2': test_df['document2'],
        'document1_id': test_df['document1_id'],
        'document2_id': test_df['document2_id'],
        'true_label': test_df['same_author_label'],
        'gram2vec_score': gram2vec_cosims,
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
        true_labels = list(test_df['same_author_label'])

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

    # For ROC curve calculations, use the DataFrame labels
    true_labels_np = np.array(test_df['same_author_label'])
    
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
    # Check if the directory exists, if not create it
    plt.savefig(f"{graph_dir}/auc_curve.png")
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

    plt.savefig(f"{graph_dir}/predicted_labels.png")