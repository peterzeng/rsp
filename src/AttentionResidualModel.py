import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class AttentionResidualDataset(Dataset):
    def __init__(self, data, model_type):
        self.data = data
        self.model_type = model_type
        print("self.model_type", self.model_type)
        self.max_length = 512
        if model_type == "rrivera1849/LUAR-MUD":
            self.tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type, local_files_only=True, trust_remote_code=True)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.model_type == "rrivera1849/LUAR-MUD" or self.model_type == "/home/zengpe/rsp/LUAR-RU":
            tokenized_text1 = self.tokenizer(item['text1'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
            tokenized_text2 = self.tokenizer(item['text2'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        else:
            tokenized_text1 = self.tokenizer(item['text1'], padding='max_length', truncation=True, max_length=self.max_length)
            tokenized_text2 = self.tokenizer(item['text2'], padding='max_length', truncation=True, max_length=self.max_length)
        
        if "residual" in item:
            label = torch.tensor(item["residual"])
        else:
            label = None
        
        if self.model_type == "rrivera1849/LUAR-MUD" or self.model_type == "/home/zengpe/rsp/LUAR-RU":
            if self.model_type == "/home/zengpe/rsp/LUAR-RU":
                if 'token_type_ids' in tokenized_text1:
                    del tokenized_text1['token_type_ids']
                if 'token_type_ids' in tokenized_text2:
                    del tokenized_text2['token_type_ids']
                
            return {
                "text1": tokenized_text1, 
                "text2": tokenized_text2, 
                "features1": item["features1"], 
                "features2": item["features2"], 
                "labels": label
                } 
        else:
            return {
                "text1": {k: v.squeeze(0).clone().detach() for k, v in tokenized_text1.items()},
                "text2": {k: v.squeeze(0).clone().detach() for k, v in tokenized_text2.items()},
                "features1": torch.tensor(item["features1"], dtype=torch.float32),
                "features2": torch.tensor(item["features2"], dtype=torch.float32),
                "labels": torch.tensor(item["residual"], dtype=torch.float32)
            }
        
class AttentionResidualModel(torch.nn.Module):
    def __init__(self, model_type, feature_dim):
        super().__init__()
        self.model_type = model_type
        
        # Change this part to load from local checkpoint
        if model_type == "/home/zengpe/rsp/LUAR-RU":
            self.enc_model = AutoModel.from_pretrained(model_type, local_files_only=True, trust_remote_code=True)
        else:
            self.enc_model = AutoModel.from_pretrained(model_type, trust_remote_code=True)

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

        self.hidden_size = 512 if self.model_type == "rrivera1849/LUAR-MUD" or self.model_type == "/home/zengpe/rsp/LUAR-RU" else self.enc_model.config.hidden_size
        self.feature_dim = feature_dim

        # Calculate embed_dim to be divisible by num_heads
        self.embed_dim = max(self.hidden_size, self.feature_dim)
        self.embed_dim = (self.embed_dim // 8) * 8  # Make it divisible by 8

        # Add LayerNorm layers
        self.ln_pre_hidden = torch.nn.LayerNorm(self.hidden_size)
        self.ln_pre_feature = torch.nn.LayerNorm(self.feature_dim)
        self.ln_post_projection = torch.nn.LayerNorm(self.embed_dim)
        # Self-attention layer
        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Projection layers to ensure all embeddings have the same dimension
        self.proj_hidden = torch.nn.Linear(self.hidden_size, self.embed_dim)
        self.proj_feature = torch.nn.Linear(self.feature_dim, self.embed_dim)
        
        # Final regression layers
        self.classification_head = self._create_classification_head(4 * self.embed_dim) 

        self.use_mean_pooling = self.model_type not in ["rrivera1849/LUAR-MUD", "/home/zengpe/rsp/LUAR-RU"]

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
    
    def forward(self, doc1, doc2, features1, features2, labels=None, layernorm=None):
        device = next(self.parameters()).device
        doc1 = {k: v.to(device) for k, v in doc1.items()}
        doc2 = {k: v.to(device) for k, v in doc2.items()}
        
        features1 = features1.to(device)
        features2 = features2.to(device)

        outputs1 = self.enc_model(**doc1)
        outputs2 = self.enc_model(**doc2)

        if self.use_mean_pooling:
            hidden1 = mean_pooling(outputs1, doc1['attention_mask'])
            hidden2 = mean_pooling(outputs2, doc2['attention_mask'])
        else:
            hidden1 = outputs1
            hidden2 = outputs2
        
        features1 = features1.squeeze(1).to(hidden1.dtype)
        features2 = features2.squeeze(1).to(hidden1.dtype)

        '''
            Doc1: g2v vector (600), model emb (700)
            Doc2: g2v vector (600), model emb (700)

            Projection: max(g2v, model emb)
            Question: When to apply layernorm?

            Option 1. Apply layernorm before projection
            Option 2. Apply layernorm after projection  
            Option 3. Apply layernorm before and after projection
        '''
        if layernorm == "pre" or layernorm == "both":
            # Apply LayerNorm before projection
            hidden1 = self.ln_pre_hidden(hidden1)
            hidden2 = self.ln_pre_hidden(hidden2)
            features1 = self.ln_pre_feature(features1)
            features2 = self.ln_pre_feature(features2)

        # Project all embeddings to the same dimension
        hidden1 = self.proj_hidden(hidden1)
        hidden2 = self.proj_hidden(hidden2)
        features1 = self.proj_feature(features1)
        features2 = self.proj_feature(features2)

        if layernorm == "post" or layernorm == "both":
            # Apply LayerNorm after projection
            hidden1 = self.ln_post_projection(hidden1)
            hidden2 = self.ln_post_projection(hidden2)
            features1 = self.ln_post_projection(features1)
            features2 = self.ln_post_projection(features2)

        # Stack all four embeddings for self-attention and apply self-attention
        stacked = torch.stack([hidden1, hidden2, features1, features2], dim=1)
        attn_output, attn_weights = self.self_attention(stacked, stacked, stacked)
        # Flatten the attention output
        flattened = attn_output.reshape(attn_output.size(0), -1)
        logits = self.classification_head(flattened)
        logits = torch.clamp(logits, min=-2.0, max=2.0)  

        if labels is not None:
            loss = self.loss_func(logits.squeeze(-1), labels.float())
            return {"loss": loss, "logits": logits, "attention_weights": attn_weights}
        else:
            return {"logits": logits, "attention_weights": attn_weights}