from collections import defaultdict
import json 
import os
import numpy as np

class BPEtokeniser():
    
    def __init__(self,text_path = '/Users/aakashrajput/MachineLearning/attention_everyone/data/input (1).txt',num_merges = 1000):
        
        self.num_merges = num_merges
        self.text = (open(text_path,'r', encoding='utf-8')).read()
        self.vocab = {}
        self.merges = []
        self.merge_ranks ={}
        
        
    
    def __get_vocab(self):
        vocab = {}
        for word in self.text.split():
            char = list(word) + ["</w>"]
            token = tuple(char)
            
            vocab[token] = vocab.get(token,0)+1
            
        return vocab
        
    
    def __get_stats(self):
        pair = defaultdict(int)
        
        for word ,freq in self.vocab.items():
            for i in range (len(word)-1):
                pair[word[i],word[i+1]] += freq
                
        
        return pair
    
    
    def __merge_vocab(self,pair,vocab):
        
        new_vocab = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        
        for word in vocab :
            word_str = " ".join(word)
            new_word = word_str.replace(bigram,replacement)
            new_word = tuple(new_word.split(" "))
            new_vocab[new_word] = vocab[word]
            
            
        return new_vocab
    
    def train(self):
        
        self.vocab = self.__get_vocab()
        
        for i in range (self.num_merges):
            pair = self.__get_stats()
            
            if not pair : break
            
            best = max(pair,key=pair.get)
            self.vocab = self.__merge_vocab(best,self.vocab)
            
            self.merges.append(best)
            
            if i % 100 == 0:
                print(f"merge {i} : {best}")
                
        self.merge_ranks = {pair : i for i , pair in enumerate(self.merges)}
        
        tokens = set()
        
        for word in self.vocab:
            for token in word:
                tokens.add(token)
                
        tokens.add("<unk>")
        
        self.token_to_id = {tok : i for i,tok in enumerate(tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        
        
    def encode_word (self,word):
        
        tokens = list(word) + ["</w>"]
        
        
        while True:
        
            pairs = [(tokens[i],tokens[i+1]) for i in range (len(tokens)-1)]
            current = None
            best_rank = float('inf')
            
            for par in pairs:
                if par in self.merge_ranks:
                    rank = self.merge_ranks[par]
                    
                    if rank < best_rank:
                        best_rank = rank
                        current = par
                    
                
            if current == None : break
            
            i = pairs.index(current)
            tokens = tokens[:i] + ["".join(current)] + tokens[i+2:]
            
        return tokens

    def encode(self,text,return_tokens = False):
        
        all_tokens = []
        
        for word in text.split():
            all_tokens.extend(self.encode_word(word))
            
        if return_tokens:
            return all_tokens
            
        return [self.token_to_id.get(tok,self.token_to_id["<unk>"]) for tok in all_tokens]
    

    
    def decode(self,ids):
        tokens = [self.id_to_token[i] for i in ids]
        text = "".join(tokens)
        return text.replace("</w>"," ")
    
    
    def save (self, path ):
        with open(path,"w") as f:
            json.dump({
                "merges" : self.merges,
                "token_to_ids" : self.token_to_id
            },f)
            
    def load(self,path):
        
        with open(path,"r") as f: data = json.load(f)
        
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.merge_ranks = {pair : i for i, pair in enumerate(self.merges)}
        
        self.token_to_id = data["token_to_ids"]
        self.id_to_token = {i : tok for tok , i in self.token_to_id.items()}
            
            
    



if __name__ == "__main__":
    
    tokenizer_path = "bpe.json"
    
    checker = BPEtokeniser()
    
  
    if os.path.exists(tokenizer_path):
        print("Loading existing tokenizer...")
        checker.load(tokenizer_path)
    else:
        print("Training tokenizer...")
        checker.train()
        checker.save(tokenizer_path)
        print("Tokenizer saved.")
    
    

    text = "I am a big winner and a nice person hahahaha asdf"

    print("\nOriginal:", text)

    tokens = checker.encode(text, return_tokens=True)
    print("Segments:", tokens)

    ids = checker.encode(text)
    print("Encoded IDs:", ids)

    decoded = checker.decode(ids)
    print("Decoded:", decoded)
            
            
            
            
            
        
            
    
    
        
            
        
        
        
        
        