import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_RE = re.compile(r"[A-Za-z0-9$%]+")
UNK_TOKEN = "<UNK>"


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def encode_text(text: str, vocab: dict):
    tokens = tokenize(text)
    unk_id = vocab.get(UNK_TOKEN, 0)
    return [vocab.get(tok, unk_id) for tok in tokens]


class NBOWClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean")
        self.fc = nn.Linear(emb_dim, 2)

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids, offsets)
        return self.fc(x)


def predict(model, vocab, text: str, device="cpu"):
    """Predict yes/no probabilities for a single text."""
    model.eval()
    token_ids = torch.tensor(encode_text(text, vocab), dtype=torch.long, device=device)
    offsets = torch.tensor([0], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(token_ids, offsets)
        probs = F.softmax(logits, dim=1)[0]
    
    yes_prob = probs[1].item()
    no_prob = probs[0].item()
    
    return {"yes": yes_prob, "no": no_prob}


def load_model(checkpoint_path: str, device="cpu"):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab = checkpoint["vocab"]
    config = checkpoint["config"]
    
    model = NBOWClassifier(
        vocab_size=len(vocab),
        emb_dim=config["emb_dim"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    return model, vocab


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_model("nbow_model_modal.pt", device=device)
    
    test_questions = [
        "Will Bitcoin reach $100,000 by the end of 2026?",
        "Will it rain tomorrow in Seattle?",
        "Will AI surpass human intelligence by 2030?",
    ]
    
    output_path = Path("inference_results.md")
    lines = [
        "# NBOW Inference Results",
        "",
        f"- **Device:** {device}",
        f"- **Vocab size:** {len(vocab)}",
        "",
        "## Predictions",
        "",
    ]

    for question in test_questions:
        result = predict(model, vocab, question, device)
        lines.append(f"- **Question:** {question}")
        lines.append(f"  YES: {result['yes']:.3f}")
        lines.append(f"  NO: {result['no']:.3f}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved inference output to: {output_path}")
