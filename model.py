import torch 
import torch.nn as nn


class attention_block(nn.Module):
    
    def __init__(self,d_model = 512):
        
        super().__init__()
        
        self.Wq = nn.Linear(d_model,d_model)
        self.Wk = nn.Linear(d_model,d_model)
        self.Wv = nn.Linear(d_model,d_model)
        self.dk = d_model
        
    
    def forward(self,x):
        
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        return (torch.softmax(Q@(K.transpose(-2,-1))/(self.dk**0.5),dim = -1))@V
    
    
class MultiHeadAttention (nn.Module):
    
    def __init__(self,d_model = 512 , total_head = 8):
        
        super().__init__()
        
        assert d_model % total_head == 0
        
        self.d_model = d_model
        self.dk = d_model//total_head
        self.total_head = total_head
        
        self.Wq = nn.Linear(d_model,d_model)
        self.Wk = nn.Linear(d_model,d_model)
        self.Wv = nn.Linear(d_model,d_model)
        
        self.Wo = nn.Linear(d_model,d_model)
    
    def forward (self,x):
        
        B,T,_ = x.shape
        
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        Q = Q.view(B,T,self.total_head,self.dk).transpose(1,2)
        K = K.view(B,T,self.total_head,self.dk).transpose(1,2)
        V = V.view(B,T,self.total_head,self.dk).transpose(1,2)
        
        atten = torch.softmax((Q@K.transpose(-2,-1)/self.dk**0.5),dim = -1)
        out = atten@V
        
        out = out.transpose(2,1).contiguous().view(B,T,self.d_model)
        
        return self.Wo(out)
    
    
class Encoder(nn.Module):
    
    pass