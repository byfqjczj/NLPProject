import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"[A-Za-z0-9$%]+")
UNK_TOKEN = "<UNK>"


@dataclass
class Example:
    text: str
    yes_prob: float
    no_prob: float


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def parse_market_examples(json_path: str) -> List[Example]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples: List[Example] = []
    skipped = 0
    for item in raw:
        name = str(item.get("name", "")).strip()
        outcomes = item.get("outcomes")
        prices = item.get("outcomePrices")
        if not name or not isinstance(outcomes, list) or not isinstance(prices, list):
            skipped += 1
            continue
        if len(outcomes) != 2 or len(prices) != 2:
            skipped += 1
            continue
        try:
            pair = {str(o).strip().lower(): float(p) for o, p in zip(outcomes, prices)}
        except (TypeError, ValueError):
            skipped += 1
            continue
        if "yes" not in pair or "no" not in pair:
            skipped += 1
            continue
        yes_prob = pair["yes"]
        no_prob = pair["no"]
        total = yes_prob + no_prob
        if total <= 0:
            skipped += 1
            continue
        yes_prob /= total
        no_prob /= total
        examples.append(Example(text=name, yes_prob=yes_prob, no_prob=no_prob))
    if not examples:
        raise ValueError("No valid yes/no training examples were found in the JSON file.")
    print(f"Loaded {len(examples)} examples from {json_path}. Skipped {skipped} invalid rows.")
    return examples


def split_examples(examples: Sequence[Example], train_ratio: float, seed: int) -> Tuple[List[Example], List[Example]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    n_train = max(1, int(len(idx) * train_ratio))
    n_train = min(n_train, len(idx) - 1)
    train = [examples[i] for i in idx[:n_train]]
    val = [examples[i] for i in idx[n_train:]]
    return train, val


def build_vocab(texts: Sequence[str], min_freq: int, max_vocab: int) -> Dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab: Dict[str, int] = {UNK_TOKEN: 0}
    sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    for token, freq in sorted_tokens:
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokenize(text)]
    if not ids:
        ids = [vocab[UNK_TOKEN]]
    return ids


class MarketDataset(Dataset):
    def __init__(self, examples: Sequence[Example], vocab: Dict[str, int]):
        self.token_ids = [encode_text(ex.text, vocab) for ex in examples]
        self.targets = [[ex.yes_prob, ex.no_prob] for ex in examples]
    def __len__(self) -> int:
        return len(self.targets)
    def __getitem__(self, idx: int):
        return self.token_ids[idx], self.targets[idx]


def collate_batch(batch):
    all_tokens: List[int] = []
    offsets: List[int] = []
    targets: List[List[float]] = []
    running_offset = 0
    for token_ids, target in batch:
        offsets.append(running_offset)
        all_tokens.extend(token_ids)
        running_offset += len(token_ids)
        targets.append(target)
    token_tensor = torch.tensor(all_tokens, dtype=torch.long)
    offset_tensor = torch.tensor(offsets, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    return token_tensor, offset_tensor, target_tensor


class NBOWClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean")
        self.fc = nn.Linear(emb_dim, 2)

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids, offsets)
        return self.fc(x)


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0
    brier = 0.0
    with torch.no_grad():
        for token_ids, offsets, targets in loader:
            token_ids = token_ids.to(device)
            offsets = offsets.to(device)
            targets = targets.to(device)
            logits = model(token_ids, offsets)
            loss = soft_cross_entropy(logits, targets)
            probs = torch.softmax(logits, dim=1)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            pred = probs.argmax(dim=1)
            true = targets.argmax(dim=1)
            correct += (pred == true).sum().item()
            brier += torch.mean((probs[:, 0] - targets[:, 0]) ** 2).item() * batch_size
    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": correct / max(1, total_count),
        "brier_yes": brier / max(1, total_count),
    }


def train(args) -> Dict[str, float]:
    examples = parse_market_examples(args.data)
    train_examples, val_examples = split_examples(examples, train_ratio=0.8, seed=args.seed)
    print(f"Train/validation split: {len(train_examples)}/{len(val_examples)} (80/20)")
    vocab = build_vocab(
        [ex.text for ex in train_examples],
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
    )
    print(f"Vocab size: {len(vocab)}")

    train_ds = MarketDataset(train_examples, vocab)
    val_ds = MarketDataset(val_examples, vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    model = NBOWClassifier(vocab_size=len(vocab), emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_val_metrics = {"loss": 0.0, "accuracy": 0.0, "brier_yes": 0.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for token_ids, offsets, targets in train_loader:
            token_ids = token_ids.to(device)
            offsets = offsets.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(token_ids, offsets)
            loss = soft_cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate(model, val_loader, device)
        final_val_metrics = val_metrics
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_brier_yes={val_metrics['brier_yes']:.4f}"
        )

    checkpoint = {
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "emb_dim": args.emb_dim,
            "min_freq": args.min_freq,
            "max_vocab": args.max_vocab,
        },
    }
    
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_out)
    print(f"Saved model checkpoint to {args.model_out}")
    print("\nSample validation predictions:")
    model.eval()
    for ex in val_examples[: min(10, len(val_examples))]:
        ids = torch.tensor(encode_text(ex.text, vocab), dtype=torch.long, device=device)
        offsets = torch.tensor([0], dtype=torch.long, device=device)
        with torch.no_grad():
            probs = torch.softmax(model(ids, offsets), dim=1)[0]
        print(
            f"- {ex.text}\n"
            f"  predicted: Yes={probs[0].item():.4f}, No={probs[1].item():.4f} | "
            f"target: Yes={ex.yes_prob:.4f}, No={ex.no_prob:.4f}"
        )

    return {
        "train_size": float(len(train_examples)),
        "val_size": float(len(val_examples)),
        "vocab_size": float(len(vocab)),
        "val_loss": final_val_metrics["loss"],
        "val_accuracy": final_val_metrics["accuracy"],
        "val_brier_yes": final_val_metrics["brier_yes"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Neural Bag-of-Words model for Yes/No market outcome probabilities.")
    parser.add_argument("--data", type=str, default="test_markets_unresolved_yes_no.json", help="Path to the training JSON file")
    parser.add_argument("--model-out", type=str, default="nbow_model.pt", help="Path to save the trained model checkpoint")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--emb-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum token frequency for vocabulary")
    parser.add_argument("--max-vocab", type=int, default=20000, help="Maximum vocabulary size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA is available")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    summary = train(args)
    print(
        "Final metrics | "
        f"val_loss={summary['val_loss']:.4f}, "
        f"val_acc={summary['val_accuracy']:.4f}, "
        f"val_brier_yes={summary['val_brier_yes']:.4f}"
    )
