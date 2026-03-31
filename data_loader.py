import os
import torch



class Dataset():
    
    def __init__(self,path = '/Users/aakashrajput/MachineLearning/attention_everyone/data/input (1).txt',block_size = 128 , split_ratio = 0.9):
        self.text = (open(path,'r', encoding='utf-8')).read()
        self.char = sorted(set(self.text))
       
        self.stoi = {ch : i for i , ch in enumerate(self.char)}
        self.itos = {i : ch for i , ch in enumerate(self.char)}
        
        self.data = torch.tensor([self.stoi[ch] for ch in self.text], dtype=torch.long)
        
        n = int(split_ratio*len(self.data))
        
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        
        self.block_size = block_size
        
    def decode (self , indices ):
        return ''.join([self.itos[int(i)] for i in indices])
    
    def encode (self,string):
        return [self.stoi[ch] for ch in string]
    
    
    def __len__(self):
        return len(self.data)
    
    def get_batch(self,batch_size = 12,split = "train",):
        
        data = self.train_data if split == "train"  else self.val_data
        
        ix = torch.randint(0,len(data) - self.block_size,(batch_size,))
        
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        
        return x,y
        
    
if __name__ == '__main__':
    
    head = Dataset()
    print(head.text[:10])
    print(head.data[:10])
    
    x , y = head.get_batch(1000)
    
    print(f"shape of x == {x.shape} and shape of y = {y.shape}")
    
    print(x[1])
    print(y[1])
    
    


