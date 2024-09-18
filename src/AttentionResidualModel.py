import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model

class AttentionResidualDataset(Dataset):
    def __init__(self, data, model_type):
        self.data = data
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type, trust_remote_code=True)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized_text1 = self.tokenizer(item['text1'], return_tensors="pt", padding='max_length', truncation=True)
        tokenized_text2 = self.tokenizer(item['text2'], return_tensors="pt", padding='max_length', truncation=True)
        label = torch.tensor(item["residual"])
        
        return {"text1": tokenized_text1, "text2": tokenized_text2, "features1": item["features1"], "features2": item["features2"], "labels": label}
    
class AttentionResidualModel(torch.nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type
        self.enc_model = AutoModel.from_pretrained(self.model_type, trust_remote_code=True)

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query","value"],
            lora_dropout=0.05,
            bias="none",
        )

        self.enc_model = get_peft_model(self.enc_model, config)
        print(self.enc_model.print_trainable_parameters())

        self.loss_func = torch.nn.MSELoss(reduction="mean")

        self.hidden_size = 512 if self.model_type == "rrivera1849/LUAR-MUD" else self.enc_model.config.hidden_size
        self.feature_dim = 680
        
        # Self-attention layer
        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=max(self.hidden_size, self.feature_dim),
            num_heads=8,
            batch_first=True
        )
        
        # Projection layers to ensure all embeddings have the same dimension
        self.proj_hidden = torch.nn.Linear(self.hidden_size, max(self.hidden_size, self.feature_dim))
        self.proj_feature = torch.nn.Linear(self.feature_dim, max(self.hidden_size, self.feature_dim))
        
        # Final regression layers
        self.classification_head = self._create_classification_head(4 * max(self.hidden_size, self.feature_dim)) 

    def _create_classification_head(self, input_size):
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(input_size // 2, input_size // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(input_size // 4, 1),
            torch.nn.Tanh()
        )
    
    def forward(self, doc1, doc2, features1, features2, labels=None):
        device = next(self.parameters()).device
        doc1 = {k: v.to(device) for k, v in doc1.items()}
        doc2 = {k: v.to(device) for k, v in doc2.items()}
        features1 = features1.to(device)
        features2 = features2.to(device)

        outputs1 = self.enc_model(**doc1)
        outputs2 = self.enc_model(**doc2)
        
        features1 = features1.squeeze(1).to(outputs1.dtype)
        features2 = features2.squeeze(1).to(outputs1.dtype)

        # Project all embeddings to the same dimension
        hidden1 = self.proj_hidden(outputs1)
        hidden2 = self.proj_hidden(outputs2)
        features1 = self.proj_feature(features1)
        features2 = self.proj_feature(features2)
        
        # Stack all four embeddings for self-attention and apply self-attention
        stacked = torch.stack([hidden1, hidden2, features1, features2], dim=1)
        attn_output, attn_weights = self.self_attention(stacked, stacked, stacked)
        # Flatten the attention output
        flattened = attn_output.reshape(attn_output.size(0), -1)
        logits = self.classification_head(flattened)

        if labels is not None:
            loss = self.loss_func(logits.squeeze(-1), labels.float())
            return {"loss": loss, "logits": logits, "attention_weights": attn_weights}
        else:
            return {"logits": logits, "attention_weights": attn_weights}