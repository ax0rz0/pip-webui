# pip-webui

Control **Pip** via an easy-to-use WebUI — with a built-in **model loader** so you
can drop in any Pip checkpoint and switch between them live from the browser.

Pip is a family of small conversational language models built from scratch (custom
**AX1** retention architecture + custom **AX0Tok** tokenizer, no PyTorch/TF, no
pretrained weights). This app runs them on **pure Python + numpy — CPU only**, so it
works on Windows / Linux / macOS with nothing but numpy installed.

## Quick start

```bash
pip install -r requirements.txt      # just numpy
python serve.py                      # then open http://localhost:8000
```

Put one or more Pip model files (`*.npz`) in the **`models/`** folder and pick one
from the dropdown in the header. The reply streams in token-by-token as Pip types.

## The model loader

- Any `*.npz` checkpoint in `models/` shows up in the **header dropdown**.
- Selecting one **loads it live** — no restart. The header shows its codename and
  shape (layers / heads / dim / context / params).
- **⟳** rescans the folder after you add a new file.
- The checkpoints are self-describing (architecture, config, and tokenizer are all
  embedded), so the loader just needs the file.

### Options

```bash
python serve.py --model models/pip4.2.npz    # load a specific model at startup
python serve.py --models-dir /path/to/models # scan a different folder
python serve.py --port 8080                  # different port
```

`--model` also accepts a path outside `models/` — it's copied in and loaded.

## Getting model files

Download Pip model files from **[huggingface.co/ax0rz0/models](https://huggingface.co/ax0rz0/models)**
and drop the `*.npz` files into the `models/` folder.

Tuning: lower the **temp** slider (~0.3–0.4) for the most coherent replies — these
are tiny models, so keep prompts light.

## Files

| file | what it is |
|---|---|
| `serve.py` | the web app + model loader |
| `infer.py` | streaming chat runtime (auto-detects arch) |
| `recurrent.py` | O(1)-per-token streaming inference |
| `ax0.py` | the AX0/AX1 model |
| `engine.py` | the from-scratch autodiff engine |
| `tokenizer.py` | byte-level BPE + `<user>/<pip>/<end>` chat protocol |
| `optim.py` | optimizer (support file) |
| `models/` | drop your `*.npz` Pip checkpoints here |
