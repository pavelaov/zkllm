# Model downloads

The real-model runs use two publicly released models. The weights are **not** redistributed in this artifact; download them with `python scripts/get_models.py` from the repository root, which also verifies the SHA-256 sums below. Please review and accept each model's licence before use.

| Model | Source | Licence | Files needed | SHA-256 (model.safetensors) |
| :--- | :--- | :--- | :--- | :--- |
| GPT-2 (124M) | huggingface.co/openai-community/gpt2 | MIT | config.json, generation_config.json, merges.txt, tokenizer.json, vocab.json, model.safetensors | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` |
| Qwen3-0.6B | huggingface.co/Qwen/Qwen3-0.6B | Apache-2.0 | config.json, generation_config.json, tokenizer.json, tokenizer_config.json, model.safetensors | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
