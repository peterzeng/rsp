from logging import config
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} "
        f"|| trainable%: {100 * trainable_params / all_param:.2f}"
    )

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class AttentionResidualDataset(Dataset):
    def __init__(self, data, model_type):
        self.data = data
        self.model_type = model_type
        # print("self.model_type", self.model_type)
        self.max_length = 512
        if model_type == "rrivera1849/LUAR-MUD":
            self.tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
        elif model_type.startswith("/"):
            # Local model path, use local_files_only
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type, local_files_only=True, trust_remote_code=True)
        else:
            # HuggingFace model, allow downloading
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type, trust_remote_code=True)
        
        # Adjust max_length for specific models
        if "mxbai-embed" in self.model_type:
            self.max_length = 512  # mxbai models support up to 4096 tokens
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different model types with appropriate tokenizer parameters
        if self.model_type == "rrivera1849/LUAR-MUD" or self.model_type == "/home/pezeng/rsp/LUAR-RU":
            tokenized_text1 = self.tokenizer(item['text1'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
            tokenized_text2 = self.tokenizer(item['text2'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        elif "mpnet" in self.model_type.lower() or "mxbai-embed" in self.model_type.lower():
            # Special handling for MPNet and MXBai models
            tokenized_text1 = self.tokenizer(item['text1'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
            tokenized_text2 = self.tokenizer(item['text2'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        else:
            # Default handling for other models
            tokenized_text1 = self.tokenizer(item['text1'], padding='max_length', truncation=True, max_length=self.max_length)
            tokenized_text2 = self.tokenizer(item['text2'], padding='max_length', truncation=True, max_length=self.max_length)
        
        if "residual" in item:
            label = torch.tensor(item["residual"])
        else:
            label = None
        
        if self.model_type == "rrivera1849/LUAR-MUD" or self.model_type == "/home/pezeng/rsp/LUAR-RU":
            if self.model_type == "/home/pezeng/rsp/LUAR-RU":
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
        elif "mpnet" in self.model_type.lower() or "mxbai-embed" in self.model_type.lower():
            # Process MPNet and MXBai-specific outputs
            processed_text1 = {}
            processed_text2 = {}
            
            for k, v in tokenized_text1.items():
                if isinstance(v, torch.Tensor):
                    processed_text1[k] = v.squeeze(0)
                else:
                    processed_text1[k] = torch.tensor(v).squeeze(0)
                    
            for k, v in tokenized_text2.items():
                if isinstance(v, torch.Tensor):
                    processed_text2[k] = v.squeeze(0)
                else:
                    processed_text2[k] = torch.tensor(v).squeeze(0)
                    
            return {
                "text1": processed_text1,
                "text2": processed_text2,
                "features1": torch.tensor(item["features1"], dtype=torch.float32),
                "features2": torch.tensor(item["features2"], dtype=torch.float32),
                "labels": torch.tensor(item["residual"], dtype=torch.float32)
            }
        elif "roberta" in self.model_type.lower() or self.model_type == "AnnaWegmann/Style-Embedding" or self.model_type == "FacebookAI/roberta-large" or self.model_type == "allenai/longformer-base-4096":
            # Process Roberta-specific outputs
            processed_text1 = {}
            processed_text2 = {}
            
            for k, v in tokenized_text1.items():
                if isinstance(v, list):
                    processed_text1[k] = torch.tensor(v)
                else:
                    processed_text1[k] = v
                    
            for k, v in tokenized_text2.items():
                if isinstance(v, list):
                    processed_text2[k] = torch.tensor(v)
                else:
                    processed_text2[k] = v
                    
            return {
                "text1": processed_text1,
                "text2": processed_text2,
                "features1": torch.tensor(item["features1"], dtype=torch.float32),
                "features2": torch.tensor(item["features2"], dtype=torch.float32),
                "labels": torch.tensor(item["residual"], dtype=torch.float32)
            }
        else:
            # Default processing for other models
            return {
                "text1": {k: v.squeeze(0).clone().detach() for k, v in tokenized_text1.items()},
                "text2": {k: v.squeeze(0).clone().detach() for k, v in tokenized_text2.items()},
                "features1": torch.tensor(item["features1"], dtype=torch.float32),
                "features2": torch.tensor(item["features2"], dtype=torch.float32),
                "labels": torch.tensor(item["residual"], dtype=torch.float32)
            }
        
class AttentionResidualModel(torch.nn.Module):
    def __init__(self, model_type, feature_dim, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.model_type = model_type
        print("Using model type:", self.model_type)
        # Change this part to load from local checkpoint
        if model_type.startswith("/"):
            # Local model path
            self.enc_model = AutoModel.from_pretrained(model_type, local_files_only=True, trust_remote_code=True)
        else:
            # HuggingFace model
            self.enc_model = AutoModel.from_pretrained(model_type, trust_remote_code=True)

        base_model_for_peft = self.enc_model
        # For sentence-transformer models, AutoModel might return a wrapper.
        # We need to get the underlying Hugging Face model for PEFT.
        if not (model_type == "/home/pezeng/rsp/LUAR-RU" or model_type == "rrivera1849/LUAR-MUD"):
            # Case 1: self.enc_model is like a SentenceTransformer (nn.Sequential)
            if isinstance(self.enc_model, torch.nn.Sequential) and \
               len(self.enc_model) > 0 and \
               hasattr(self.enc_model[0], 'auto_model'):
                base_model_for_peft = self.enc_model[0].auto_model
            # Case 2: self.enc_model is like sentence_transformers.models.Transformer (has .auto_model)
            # and is not already a raw Hugging Face model (which wouldn't typically have .auto_model unless it's the same object)
            # and also ensure it's not already a PeftModel.
            elif hasattr(self.enc_model, 'auto_model') and \
                 self.enc_model.auto_model is not self.enc_model and \
                 "PeftModel" not in str(type(self.enc_model)):
                base_model_for_peft = self.enc_model.auto_model
        
        # Find all available module names for debugging
        available_modules = [name for name, _ in base_model_for_peft.named_modules()]
        # print("Available modules in the model:", available_modules)
        
        # Try to determine the correct attention module names by examining the model structure
        if hasattr(base_model_for_peft, "config") and hasattr(base_model_for_peft.config, "model_type"):
            model_type_from_config = base_model_for_peft.config.model_type
            print(f"Model type from config: {model_type_from_config}")
            
            # Try to automatically detect attention modules by examining common patterns
            attention_query_modules = [name for name in available_modules if 
                                      ('query' in name.lower() or '.q' in name.lower()) and 
                                      ('attention' in name.lower() or 'attn' in name.lower())]
            
            attention_value_modules = [name for name in available_modules if 
                                      ('value' in name.lower() or '.v' in name.lower()) and 
                                      ('attention' in name.lower() or 'attn' in name.lower())]
            
            if attention_query_modules and attention_value_modules:
                # Use the first matching module names as they're likely to be consistent across layers
                target_modules = []
                # Extract common pattern from the module names
                if '.' in attention_query_modules[0]:
                    query_suffix = attention_query_modules[0].split('.')[-2] + '.' + attention_query_modules[0].split('.')[-1]
                    value_suffix = attention_value_modules[0].split('.')[-2] + '.' + attention_value_modules[0].split('.')[-1]
                    target_modules = [query_suffix, value_suffix]
                else:
                    target_modules = [attention_query_modules[0], attention_value_modules[0]]
                
                print(f"Automatically detected target modules: {target_modules}")
            # Set target modules based on known model architectures
            elif model_type_from_config == "mpnet":
                target_modules = ["attention.attn.q", "attention.attn.v"]
            elif model_type_from_config == "bert":
                target_modules = ["attention.self.query", "attention.self.value"]
            elif model_type_from_config == "roberta":
                target_modules = ["attention.self.query", "attention.self.value"]
            elif "mxbai" in self.model_type.lower():
                # Common attention module patterns for E5 family models which MXBai is based on
                target_modules = ["attention.self.query", "attention.self.value"]
            else:
                # Default fallback
                target_modules = ["query", "value"]
        else:
            # Fallback if we can't determine from config
            target_modules = ["query", "value"]
            
        print(f"Using target modules for LoRA: {target_modules}")
            
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )

        self.enc_model = get_peft_model(base_model_for_peft, config)
        print(self.enc_model.print_trainable_parameters())
        # print_trainable_parameters(self.enc_model)

        self.loss_func = torch.nn.MSELoss(reduction="mean")

        # Get hidden size from model config
        if hasattr(self.enc_model, 'config') and hasattr(self.enc_model.config, 'hidden_size'):
            self.hidden_size = self.enc_model.config.hidden_size
        else:
            self.hidden_size = 512 if self.model_type == "rrivera1849/LUAR-MUD" or self.model_type == "/home/pezeng/rsp/LUAR-RU" else 768
        
        print(f"Using hidden size: {self.hidden_size}")
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

        self.use_mean_pooling = self.model_type not in ["rrivera1849/LUAR-MUD", "/home/pezeng/rsp/LUAR-RU"]

        # Increase dropouts throughout
        # self.dropout = torch.nn.Dropout(dropout_rate)

    def _create_classification_head(self, input_size):
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),  # Higher dropout in classification head
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

        if "mxbai-embed" in self.model_type.lower():
            # For mxbai models, use CLS pooling as recommended in the docs
            hidden1 = outputs1.last_hidden_state[:, 0]  # CLS token
            hidden2 = outputs2.last_hidden_state[:, 0]  # CLS token
            # Apply normalization for mxbai models
            hidden1 = F.normalize(hidden1, p=2, dim=1)
            hidden2 = F.normalize(hidden2, p=2, dim=1)
        elif self.use_mean_pooling:
            hidden1 = mean_pooling(outputs1, doc1['attention_mask'])
            hidden2 = mean_pooling(outputs2, doc2['attention_mask'])
            # Add normalization only for mpnet models
            if "mpnet" in self.model_type.lower():
                hidden1 = F.normalize(hidden1, p=2, dim=1)
                hidden2 = F.normalize(hidden2, p=2, dim=1)
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

        # Apply dropout in forward pass
        # hidden1 = self.dropout(hidden1)
        # hidden2 = self.dropout(hidden2)
        # features1 = self.dropout(features1)
        # features2 = self.dropout(features2)

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