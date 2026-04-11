from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from full_arch import FullTransformer
from train import BOS, EOS, pick_device
from bpe import BPETokeniser

_PROJECT_ROOT = Path(__file__).resolve().parent


def _first_nonempty_line(path: Path) -> str | None:
    """First non-empty line after strip, or None if file missing / only blanks."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s:
            return s
    return None


def resolve_checkpoint(path: Path) -> Path:
    """Use explicit path if present; else checkpoint.pt in project root; else newest *.pt."""
    p = Path(path)
    if p.is_file():
        return p
    default_ckpt = _PROJECT_ROOT / "checkpoint.pt"
    if default_ckpt.is_file():
        print(f"No file at {path}; using {default_ckpt}")
        return default_ckpt
    candidates = sorted(_PROJECT_ROOT.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if candidates:
        print(f"No file at {path}; using newest: {candidates[0]}")
        return candidates[0]
    raise FileNotFoundError(
        f"Checkpoint not found: {path}. Train first or pass --checkpoint path/to.pt"
    )


def load_model(ckpt_path: Path, device: torch.device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    vocab = BPETokeniser.from_itos(ckpt["vocab_itos"], ckpt.get("vocab_merges", []))
    model = FullTransformer(
        src_vocab_size=cfg["vocab_size"],
        tgt_vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        total_head=cfg["num_heads"],
        dropout=cfg["dropout"],
        max_len=max(cfg["max_len"] + 16, 512),
        share_embeddings=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, cfg


@torch.inference_mode()
def greedy_decode(
    model: FullTransformer,
    vocab: BPETokeniser,
    src_text: str,
    max_len: int,
    device: torch.device,
) -> str:
    body = vocab.encode(src_text)[: max_len - 1]
    src_ids = body + [EOS]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    # Encode source once — reuse for every decoding step
    memory = model.encode(src)

    ys = [BOS]
    for _ in range(max_len + 4):
        tgt = torch.tensor([ys], dtype=torch.long, device=device)
        logits = model.decode(tgt, memory)
        next_id = int(logits[0, -1].argmax().item())
        ys.append(next_id)
        if next_id == EOS:
            break
    return vocab.decode(ys)


@torch.inference_mode()
def beam_decode(
    model: FullTransformer,
    vocab: BPETokeniser,
    src_text: str,
    max_len: int,
    device: torch.device,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> str:
    body = vocab.encode(src_text)[: max_len - 1]
    src_ids = body + [EOS]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    V = len(vocab.itos)

    # Encode source ONCE — reused for all beam steps
    memory = model.encode(src)

    beams: list[tuple[float, list[int]]] = [(0.0, [BOS])]
    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_len + 4):
        candidates: list[tuple[float, list[int]]] = []
        for score, seq in beams:
            if seq[-1] == EOS:
                completed.append((score, seq))
                continue
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            # Decoder only — memory already computed
            logits = model.decode(tgt, memory)
            logp = F.log_softmax(logits[0, -1], dim=-1)
            topk_logp, topk_id = torch.topk(logp, min(beam_size, V))
            for lp, tid in zip(topk_logp.tolist(), topk_id.tolist()):
                candidates.append((score + lp, seq + [tid]))

        if not candidates:
            break

        def norm(s: float, seq: list[int]) -> float:
            L = max(len(seq) - 1, 1)
            return s / (L**length_penalty)

        candidates.sort(key=lambda x: norm(x[0], x[1]), reverse=True)
        beams = candidates[:beam_size]

        if all(s[-1] == EOS for _, s in beams):
            completed.extend(beams)
            break

    if not completed:
        completed = beams

    best = max(completed, key=lambda x: x[0] / max(len(x[1]) - 1, 1) ** length_penalty)
    return vocab.decode(best[1])


def parse_args():
    p = argparse.ArgumentParser(description="Transformer inference")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_PROJECT_ROOT / "checkpoint.pt",
        help="Defaults to checkpoint.pt in this folder; falls back to any .pt here if missing.",
    )
    p.add_argument("--text", type=str, default="", help="Source string to copy / translate.")
    p.add_argument(
        "--prompt_file",
        type=Path,
        default=None,
        help="Read first line as source. If unset and --text empty, uses inp.txt line 1.",
    )
    p.add_argument("--beam_size", type=int, default=1, help="1 = greedy; 4 matches paper beam.")
    p.add_argument("--length_penalty", type=float, default=0.6)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (CUDA>MPS>CPU), mps (Apple GPU), cuda, or cpu.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    ckpt_path = resolve_checkpoint(args.checkpoint)
    model, vocab, cfg = load_model(ckpt_path, device)
    max_len = cfg["max_len"]

    inp_default = _PROJECT_ROOT / "inp.txt"
    if args.prompt_file is not None:
        pf = Path(args.prompt_file)
        src = _first_nonempty_line(pf) or ""
        if not src:
            raise SystemExit(
                f"No non-empty lines in {pf} (missing or empty). Add text or use --text."
            )
    else:
        src = args.text.strip()
        if not src:
            src = _first_nonempty_line(inp_default) or ""
        if not src:
            src = "hello"

    if args.beam_size <= 1:
        out = greedy_decode(model, vocab, src, max_len, device)
    else:
        out = beam_decode(
            model,
            vocab,
            src,
            max_len,
            device,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
        )

    print("src:", repr(src))
    print("out:", repr(out))


if __name__ == "__main__":
    main()
