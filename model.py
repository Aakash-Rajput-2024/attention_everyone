import math
import torch
import torch.nn as nn


class attention_block(nn.Module):


    def __init__(self, d_model=512):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.dk = d_model

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        return (torch.softmax(Q @ (K.transpose(-2, -1)) / (self.dk**0.5), dim=-1)) @ V


class SinusoidalPositionalEncoding(nn.Module):


    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :].to(x.dtype)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, total_head=8, dropout=0.0):
        super().__init__()
        assert d_model % total_head == 0

        self.d_model = d_model
        self.dk = d_model // total_head
        self.total_head = total_head

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

      
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, x_en=None):
        
        B, Tq, _ = x.shape

        if x_en is None:
           
            Q = self.Wq(x).view(B, Tq, self.total_head, self.dk).transpose(1, 2)
            K = self.Wk(x).view(B, Tq, self.total_head, self.dk).transpose(1, 2)
            V = self.Wv(x).view(B, Tq, self.total_head, self.dk).transpose(1, 2)
        else:
         
            Tk = x_en.shape[1]
            Q = self.Wq(x).view(B, Tq, self.total_head, self.dk).transpose(1, 2)
            K = self.Wk(x_en).view(B, Tk, self.total_head, self.dk).transpose(1, 2)
            V = self.Wv(x_en).view(B, Tk, self.total_head, self.dk).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / self.dk**0.5
        atten = self.attn_drop(torch.softmax(scores, dim=-1))
        out = atten @ V

        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.Wo(out)


class MaskedMultiHeadAttention(nn.Module):


    def __init__(self, d_model=512, total_head=8, dropout=0.0):
        super().__init__()
        assert d_model % total_head == 0

        self.d_model = d_model
        self.dk = d_model // total_head
        self.total_head = total_head

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
       
        B, T, _ = x.shape

        Q = self.Wq(x).view(B, T, self.total_head, self.dk).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.total_head, self.dk).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.total_head, self.dk).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / self.dk**0.5
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=scores.dtype))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        atten = self.attn_drop(torch.softmax(scores, dim=-1))
        out = atten @ V

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.Wo(out)


class Encoder(nn.Module):

    def __init__(self, d_model=512, total_head=8, dropout=0.1):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, total_head, dropout=dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
       
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x + self.dropout(self.MHA(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class DecoderLayer(nn.Module):
   

    def __init__(self, d_model=512, total_head=8, dropout=0.1):
        super().__init__()
        self.MMHA = MaskedMultiHeadAttention(d_model, total_head, dropout=dropout)
        self.MHA = MultiHeadAttention(d_model, total_head, dropout=dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_en=None):

        x = x + self.dropout(self.MMHA(self.ln1(x)))
        x = x + self.dropout(self.MHA(self.ln2(x), x_en))
        x = x + self.dropout(self.ffn(self.ln3(x)))
        return x


class Decoder(nn.Module):
  

    def __init__(self, d_model=512, vocab_size=1000, total_head=8, dropout=0.1):
        super().__init__()
        self.layer = DecoderLayer(d_model, total_head, dropout=dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, x_en=None):
        return self.fc_out(self.layer(x, x_en))
