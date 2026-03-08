import json
from pathlib import Path
from types import SimpleNamespace

import modal


APP_NAME = "nlpproject-nbow-trainer"
VOLUME_NAME = "nlpproject-nbow-data"
VOLUME_PATH = "/data"
TRAIN_SCRIPT_PATH = "/root/train_nbow.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1")
    .add_local_file("train_nbow.py", remote_path=TRAIN_SCRIPT_PATH)
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _load_train_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_nbow", TRAIN_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training module from {TRAIN_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=600)
def upload_dataset(dataset_text: str, remote_name: str = "test_markets_unresolved_yes_no.json") -> str:
    remote_path = Path(VOLUME_PATH) / remote_name
    remote_path.write_text(dataset_text, encoding="utf-8")
    volume.commit()
    return str(remote_path)


@app.function(image=image, volumes={VOLUME_PATH: volume}, gpu="T4", timeout=3600)
def train_remote_gpu(
    data_path: str,
    model_out: str,
    epochs: int,
    batch_size: int,
    emb_dim: int,
    lr: float,
    weight_decay: float,
    min_freq: int,
    max_vocab: int,
    seed: int,
):
    train_nbow = _load_train_module()

    args = SimpleNamespace(
        data=data_path,
        model_out=model_out,
        epochs=epochs,
        batch_size=batch_size,
        emb_dim=emb_dim,
        lr=lr,
        weight_decay=weight_decay,
        min_freq=min_freq,
        max_vocab=max_vocab,
        seed=seed,
        cpu=False,
    )
    summary = train_nbow.train(args)
    volume.commit()
    return summary


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=3600)
def train_remote_cpu(
    data_path: str,
    model_out: str,
    epochs: int,
    batch_size: int,
    emb_dim: int,
    lr: float,
    weight_decay: float,
    min_freq: int,
    max_vocab: int,
    seed: int,
):
    train_nbow = _load_train_module()

    args = SimpleNamespace(
        data=data_path,
        model_out=model_out,
        epochs=epochs,
        batch_size=batch_size,
        emb_dim=emb_dim,
        lr=lr,
        weight_decay=weight_decay,
        min_freq=min_freq,
        max_vocab=max_vocab,
        seed=seed,
        cpu=True,
    )
    summary = train_nbow.train(args)
    volume.commit()
    return summary


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=600)
def download_artifact(remote_path: str) -> bytes:
    return Path(remote_path).read_bytes()


@app.local_entrypoint()
def main(
    local_data_path: str = "test_markets_unresolved_yes_no.json",
    local_model_out: str = "nbow_model.pt",
    remote_data_name: str = "test_markets_unresolved_yes_no.json",
    remote_model_name: str = "nbow_model.pt",
    use_gpu: bool = True,
    epochs: int = 20,
    batch_size: int = 32,
    emb_dim: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    min_freq: int = 1,
    max_vocab: int = 20000,
    seed: int = 42,
):
    data_text = Path(local_data_path).read_text(encoding="utf-8")

    remote_data_path = upload_dataset.remote(data_text, remote_data_name)
    # Use string concatenation for remote Linux path (avoid Windows Path backslashes)
    remote_model_path = f"{VOLUME_PATH}/{remote_model_name}"

    train_kwargs = {
        "data_path": remote_data_path,
        "model_out": remote_model_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "emb_dim": emb_dim,
        "lr": lr,
        "weight_decay": weight_decay,
        "min_freq": min_freq,
        "max_vocab": max_vocab,
        "seed": seed,
    }

    if use_gpu:
        summary = train_remote_gpu.remote(**train_kwargs)
    else:
        summary = train_remote_cpu.remote(**train_kwargs)

    model_bytes = download_artifact.remote(remote_model_path)
    Path(local_model_out).write_bytes(model_bytes)

    results_file = "training_results.md"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("# NBOW Training Results\n\n")
        f.write(f"**Model saved to:** `{local_model_out}`\n\n")
        
        f.write("## Training Configuration\n\n")
        f.write(f"- **Device:** {'GPU (T4)' if use_gpu else 'CPU'}\n")
        f.write(f"- **Epochs:** {epochs}\n")
        f.write(f"- **Batch Size:** {batch_size}\n")
        f.write(f"- **Embedding Dimension:** {emb_dim}\n")
        f.write(f"- **Learning Rate:** {lr}\n")
        f.write(f"- **Weight Decay:** {weight_decay}\n")
        f.write(f"- **Min Frequency:** {min_freq}\n")
        f.write(f"- **Max Vocab Size:** {max_vocab}\n")
        f.write(f"- **Random Seed:** {seed}\n\n")
        
        f.write("## Dataset\n\n")
        f.write(f"- **Training Size:** {int(summary['train_size'])} examples\n")
        f.write(f"- **Validation Size:** {int(summary['val_size'])} examples\n")
        f.write(f"- **Vocabulary Size:** {int(summary['vocab_size'])} tokens\n\n")
        
        f.write("## Final Validation Metrics\n\n")
        f.write(f"- **Validation Loss:** {summary['val_loss']:.4f}\n")
        f.write(f"- **Validation Accuracy:** {summary['val_accuracy']:.4f} ({summary['val_accuracy']*100:.2f}%)\n")
        
        f.write("---\n")
        f.write(f"*Training completed on Modal*\n")
    
    print(f"Training finished on Modal.")
    print(f"Model saved to: {local_model_out}")
    print(f"Results saved to: {results_file}")
