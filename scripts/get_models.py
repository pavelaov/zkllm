"""Download the model files the paper's real-model runs need (no weights are redistributed).

Downloads GPT-2 (openai-community/gpt2, MIT) and Qwen3-0.6B (Qwen/Qwen3-0.6B, Apache-2.0) from the Hugging Face hub and verifies the SHA-256 sums in models/SHA256SUMS.txt. Run:  python scripts/get_models.py            (from the repository root)
"""
import hashlib
import os
import sys
import urllib.request

FILES = {
    # (url, destination, expected sha256 or None)
    "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors":
        ("models/gpt2/model.safetensors",
         "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"),
    "https://huggingface.co/openai-community/gpt2/resolve/main/config.json":
        ("models/gpt2/config.json", None),
    "https://huggingface.co/openai-community/gpt2/resolve/main/generation_config.json":
        ("models/gpt2/generation_config.json", None),
    "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt":
        ("models/gpt2/merges.txt", None),
    "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json":
        ("models/gpt2/tokenizer.json", None),
    "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json":
        ("models/gpt2/vocab.json", None),
    "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors":
        ("models/qwen3-0.6b/model.safetensors",
         "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
    "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json":
        ("models/qwen3-0.6b/config.json", None),
    "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/generation_config.json":
        ("models/qwen3-0.6b/generation_config.json", None),
    "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json":
        ("models/qwen3-0.6b/tokenizer.json", None),
    "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer_config.json":
        ("models/qwen3-0.6b/tokenizer_config.json", None),
}

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ok = True
    for url, (rel, want) in FILES.items():
        dest = os.path.join(root, rel.replace("/", os.sep))
        if os.path.isfile(dest) and (want is None or sha256(dest) == want):
            print(f"ok       {rel}")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"download {rel} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED   {rel}: {exc}")
            ok = False
            continue
        if want is not None:
            got = sha256(dest)
            if got != want:
                print(f"CHECKSUM MISMATCH {rel}: {got} != {want}")
                ok = False
            else:
                print(f"verified {rel}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
