from __future__ import annotations

from collections import defaultdict
import json

# Fixed special token IDs — must match train.py constants
PAD_TOKEN = "<pad>"  # 0
BOS_TOKEN = "<bos>"  # 1
EOS_TOKEN = "<eos>"  # 2
UNK_TOKEN = "<unk>"  # 3
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD, BOS, EOS, UNK = 0, 1, 2, 3


class BPETokeniser:


    def __init__(self, num_merges: int = 1000):
        self.num_merges = num_merges
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}

    def train(self, text: str) -> None:

        vocab = self._get_vocab(text)
        for i in range(self.num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            self.merges.append(best)
            if i % 100 == 0:
                print(f"  merge {i:4d}: {''.join(best)!r}")
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._build_vocab(vocab)

    def _get_vocab(self, text: str) -> dict[tuple[str, ...], int]:
        vocab: dict[tuple[str, ...], int] = {}
        for word in text.split():
            token = tuple(list(word) + ["</w>"])
            vocab[token] = vocab.get(token, 0) + 1
        return vocab

    def _get_stats(self, vocab: dict) -> dict[tuple[str, str], int]:
        pairs: dict[tuple[str, str], int] = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: tuple[str, str], vocab: dict) -> dict:
        bigram = " ".join(pair)
        replacement = "".join(pair)
        new_vocab = {}
        for word in vocab:
            new_word = tuple(" ".join(word).replace(bigram, replacement).split())
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def _build_vocab(self, vocab: dict) -> None:

        tokens: set[str] = set()
        for word in vocab:
            for tok in word:
                tokens.add(tok)

        self.itos = SPECIAL_TOKENS + sorted(tokens - set(SPECIAL_TOKENS))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}



    def _encode_word(self, word: str) -> list[str]:
       
        tokens = list(word) + ["</w>"]
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best_rank, current = float("inf"), None
            for pair in pairs:
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank, current = rank, pair
            if current is None:
                break
            i = pairs.index(current)
            tokens = tokens[:i] + ["".join(current)] + tokens[i + 2:]
        return tokens

    def encode(self, text: str) -> list[int]:
    
        ids: list[int] = []
        for word in text.split():
            for tok in self._encode_word(word):
                ids.append(self.stoi.get(tok, UNK))
        return ids

    def decode(self, ids: list[int]) -> str:

        parts: list[str] = []
        for i in ids:
            if i in (PAD, BOS):
                continue
            if i == EOS:
                break
            if 0 <= i < len(self.itos):
                tok = self.itos[i]
                if tok in SPECIAL_TOKENS:
                    continue
                parts.append(tok)

        return "".join(parts).replace("</w>", " ").strip()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "merges": list(self.merges),
                "itos": self.itos,
                "num_merges": self.num_merges,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.merges = [tuple(p) for p in data["merges"]]
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        if "itos" in data:
            # New format — direct load
            self.itos = data["itos"]
        else:
            # Old format stored "token_to_ids" without guaranteed special token positions.
            # Rebuild itos so specials are fixed at 0-3.
            print("  [bpe] Old bpe.json format detected — rebuilding itos with correct special tokens.")
            old_tokens = set(data.get("token_to_ids", {}).keys())
            subword_tokens = sorted(old_tokens - set(SPECIAL_TOKENS))
            self.itos = SPECIAL_TOKENS + subword_tokens

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.num_merges = data.get("num_merges", len(self.merges))



    @classmethod
    def from_itos(cls, itos: list[str], merges: list | None = None) -> "BPETokeniser":
        """Reconstruct vocab from a checkpoint's itos + merges lists."""
        obj = cls.__new__(cls)
        obj.itos = list(itos)
        obj.stoi = {tok: i for i, tok in enumerate(obj.itos)}
        obj.merges = [tuple(p) for p in (merges or [])]
        obj.merge_ranks = {pair: i for i, pair in enumerate(obj.merges)}
        obj.num_merges = len(obj.merges)
        return obj

    def __len__(self) -> int:
        return len(self.itos)
if __name__ == "__main__":
    import os

    tokenizer_path = "bpe.json"
    data_path = "data/input (1).txt"

    tok = BPETokeniser(num_merges=1000)

    if os.path.exists(tokenizer_path):
        print("Loading existing tokenizer...")
        tok.load(tokenizer_path)
    else:
        print(f"Training tokenizer on {data_path} ...")
        tok.train(open(data_path, encoding="utf-8").read())
        tok.save(tokenizer_path)
        print("Tokenizer saved.")

    text = "To be or not to be that is the question"
    print("\nOriginal :", text)
    print("Segments :", [tok.itos[i] for i in tok.encode(text)])
    print("IDs      :", tok.encode(text))
    print("Decoded  :", tok.decode(tok.encode(text)))
    print(f"Vocab size: {len(tok):,}")