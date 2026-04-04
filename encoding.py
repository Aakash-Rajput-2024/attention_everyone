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
    
    def pos_encode(self,text):
        
        ids = self.bpe.encode(text)
        seq_len = len(ids)
        
        PE = torch.zeros((seq_len, self.d_model), device=self.embedding.weight.device)
        
        for pos in range (seq_len):
            for i in range(0,self.d_model,2):
                PE[pos,i] = torch.sin(pos/(10000**(i/self.d_model)))
                if i+1 < self.d_model :PE[pos,i+1] = torch.cos(pos/(10000**(i/self.d_model)))
        
        return PE
    
    def get_embeddings(self,text):
        
        ids = torch.tensor(self.bpe.encode(text), dtype=torch.long)
        PE = self.pos_encode(text)
        EM = self.forward(ids)
        
        return PE + EM
    
    
    
        
        
        
        
        
        
        

