import torch 
import torch.nn as nn
from bpe import BPEtokeniser
import os
import numpy as np



class Embedding(nn.Module):
    
    def __init__(self,vocab_size,d_model = 64,tokenizer_path = '/Users/aakashrajput/MachineLearning/attention_everyone/bpe.json'):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.d_model = d_model
        self.bpe = BPEtokeniser()
        
        if os.path.exists(tokenizer_path):
            print("Loading existing tokenizer...")
            self.bpe.load(tokenizer_path)
        else:
            print("Training tokenizer...")
            self.bpe.train()
            self.bpe.save(tokenizer_path)
            print("Tokenizer saved.")
            
        
    def forward(self,ids):
        return self.embedding(ids)
    
    def pos_encoding(self,seq_len):
        
        device = self.embedding.weight.device
        d_model = self.d_model
        
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        PE = torch.zero((seq_len,d_model),device = device)
        
        PE[:,0::2] = torch.sin(position*div_term)
        PE[:,1::2] = torch.cos(position*div_term)
        
        return PE
    
    def get_embeddings(self,text):
        
        ids = torch.tensor(self.bpe.encode(text), dtype=torch.long)
        PE = self.pos_encode(len(ids))
        EM = self.forward(ids)
        
        return PE + EM
    
    #todo : more opti on the PE part 
    
    
    
        
        
        
        
        
        
        

