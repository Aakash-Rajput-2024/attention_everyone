from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from full_arch import FullTransformer
from bpe import BPETokeniser

PAD, BOS, EOS, UNK = 0, 1, 2, 3


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def pick_device(kind: str = "auto") -> torch.device:
    k = kind.lower().strip()
    if k == "cpu":
        return torch.device("cpu")
    if k == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if k == "mps":
        if not _mps_available():
            raise RuntimeError(
                "MPS requested but not available (need Apple Silicon Mac and a PyTorch build with MPS)."
            )
        return torch.device("mps")
    if k != "auto":
        raise ValueError(f"Unknown --device {kind!r}; use auto, cpu, cuda, or mps.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


class CopyLineDataset(Dataset):

    def __init__(self, lines: list[str], max_len: int):
        self.lines = [ln.strip() for ln in lines if ln.strip()]
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx: int):
        return self.lines[idx][: self.max_len]


def collate_batch(
    batch: list[str],
    vocab: BPETokeniser,
    max_src: int,
    max_tgt: int,
    device: torch.device,
):
    src_list: list[list[int]] = []
    dec_in_list: list[list[int]] = []
    lab_list: list[list[int]] = []

    for line in batch:
        body = vocab.encode(line)[: max_tgt - 2]
        src_ids = body + [EOS]
        tgt_full = [BOS] + body + [EOS]
        dec_in_list.append(tgt_full[:-1])
        lab_list.append(tgt_full[1:])
        src_list.append(src_ids)

    def pad2d(rows: list[list[int]], cap: int, pad: int = PAD):
        out = torch.full((len(rows), cap), pad, dtype=torch.long)
        for i, r in enumerate(rows):
            L = min(len(r), cap)
            out[i, :L] = torch.tensor(r[:L], dtype=torch.long)
        return out

    max_s = min(max_src, max(len(x) for x in src_list))
    max_t = min(max_tgt, max(len(x) for x in dec_in_list))

    src = pad2d(src_list, max_s).to(device)
    dec_in = pad2d(dec_in_list, max_t).to(device)
    labels = pad2d(lab_list, max_t, pad=-100).to(device)
    return src, dec_in, labels


def noam_lr(step: int, d_model: int, warmup: int) -> float:
    step = max(step, 1)
    return (d_model**-0.5) * min(step**-0.5, step * (warmup**-1.5))


def parse_args():
    p = argparse.ArgumentParser(description="Transformer training (paper-style hyperparameters)")
    p.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "input (1).txt",
        help="Text file; each line is one example. Default: Shakespeare corpus.",
    )
    p.add_argument(
        "--bpe_vocab",
        type=Path,
        default=Path(__file__).resolve().parent / "bpe.json",
        help="Path to save/load the trained BPE vocabulary.",
    )
    p.add_argument(
        "--num_merges",
        type=int,
        default=1000,
        help="Number of BPE merge operations to train (only used when building vocab fresh).",
    )
    p.add_argument(
        "--retrain_bpe",
        action="store_true",
        help="Force retrain BPE vocab even if bpe.json already exists.",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=4000)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint.pt",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint and train from scratch (still saves to --checkpoint).",
    )
    p.add_argument("--tiny", action="store_true", help="Small model for quick CPU tests.")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (CUDA>MPS>CPU), mps (Apple GPU), cuda, or cpu.",
    )
    return p.parse_args()


def _load_checkpoint_file(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)

    ckpt_path = Path(args.checkpoint)
    resume = ckpt_path.is_file() and not args.no_resume
    if ckpt_path.is_file() and args.no_resume:
        print(
            f"Note: {ckpt_path} exists; --no-resume — training from scratch "
            f"(checkpoint will be overwritten on exit)."
        )

    text = Path(args.data).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    split = int(0.9 * len(lines)) if len(lines) > 1 else len(lines)
    train_lines = lines[:split] if split > 0 else lines

    # ------------------------------------------------------------------
    # BPE vocabulary: train once on the corpus, reuse across training runs.
    # The vocab is saved to bpe.json separately from the model checkpoint.
    # ------------------------------------------------------------------
    bpe_path = Path(args.bpe_vocab)
    vocab = BPETokeniser(num_merges=args.num_merges)

    if bpe_path.is_file() and not args.retrain_bpe:
        print(f"Loading BPE vocab from {bpe_path} ...")
        vocab.load(str(bpe_path))
        print(f"  vocab_size={len(vocab):,}  merges={len(vocab.merges):,}")
    else:
        print(f"Training BPE ({args.num_merges} merges) on {args.data} ...")
        vocab.train(text)
        vocab.save(str(bpe_path))
        print(f"BPE vocab saved → {bpe_path}  vocab_size={len(vocab):,}")

    if resume:
        ckpt = _load_checkpoint_file(ckpt_path, device)
        cfg = ckpt["config"]
        # Always reconstruct vocab from the checkpoint to guarantee ID consistency
        vocab = BPETokeniser.from_itos(ckpt["vocab_itos"], ckpt.get("vocab_merges", []))
        vocab_size = cfg["vocab_size"]
        d_model = cfg["d_model"]
        num_layers = cfg["num_layers"]
        num_heads = cfg["num_heads"]
        dropout = cfg["dropout"]
        max_len = cfg["max_len"]
        global_step = int(ckpt.get("global_step", 0))
        epoch = int(ckpt.get("epoch", 0))
    else:
        if args.tiny:
            args.d_model = 128
            args.num_layers = 2
            args.num_heads = 4
            args.batch_size = min(args.batch_size, 8)
        ckpt = None
        vocab_size = len(vocab)
        d_model = args.d_model
        num_layers = args.num_layers
        num_heads = args.num_heads
        dropout = args.dropout
        max_len = args.max_len
        global_step = 0
        epoch = 0

    ds = CopyLineDataset(train_lines, max_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: b,
    )

    model = FullTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        total_head=num_heads,
        dropout=dropout,
        max_len=max(max_len + 16, 512),
        share_embeddings=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-7,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    if resume and ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)

    if resume:
        print(
            f"Resuming from {ckpt_path} | epoch={epoch} global_step={global_step}\n"
            f"Device: {device} | vocab={vocab_size} (BPE) | d_model={d_model} | "
            f"layers={num_layers} | heads={num_heads}"
        )
    else:
        print(
            f"Device: {device} | vocab={vocab_size} (BPE) | d_model={d_model} | "
            f"layers={num_layers} | heads={num_heads}"
        )
    print(
        f"Data: {args.data}\n"
        f"Adam β=(0.9,0.98) ε=1e-9 | label_smoothing={args.label_smoothing} | "
        f"warmup={args.warmup_steps} | grad_clip={args.grad_clip}\n"
        f"Ctrl+C to stop and save checkpoint → {args.checkpoint}"
    )

    def save_checkpoint():
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab_itos": vocab.itos,
            "vocab_merges": vocab.merges,   # needed to rebuild merge_ranks on resume
            "config": {
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "dropout": dropout,
                "max_len": max_len,
                "vocab_size": vocab_size,
            },
            "global_step": global_step,
            "epoch": epoch,
        }
        torch.save(payload, args.checkpoint)
        print(f"Saved checkpoint ({args.checkpoint}).")

    try:
        while True:
            epoch += 1
            running_loss = 0.0
            n_batches = 0
            tokens_processed = 0
            t_epoch = time.perf_counter()
            model.train()

            for batch_lines in loader:
                src, dec_in, labels = collate_batch(
                    batch_lines,
                    vocab,
                    max_len,
                    max_len + 2,
                    device,
                )

                tokens_processed += int((src != PAD).sum().item())
                tokens_processed += int((dec_in != PAD).sum().item())

                global_step += 1
                lr = noam_lr(global_step, d_model, args.warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                logits = model(src, dec_in)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            elapsed = time.perf_counter() - t_epoch
            mean_loss = running_loss / max(n_batches, 1)
            tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}  mean_loss={mean_loss:.6f}  "
                f"lr={cur_lr:.2e}  step={global_step}  "
                f"tok/s={tok_per_sec:,.0f}  ({tokens_processed:,} tok in {elapsed:.2f}s)"
            )

    except KeyboardInterrupt:
        print("\nKeyboard interrupt — stopping.")
    finally:
        save_checkpoint()


if __name__ == "__main__":
    main()
