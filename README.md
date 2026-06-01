# Ka-myii 🎨✨

**Automated VTuber Model Studio**

Ka-myii is a local, self-hosted web app that turns a text prompt into a
**rig-ready Live2D character**. It generates high-quality anime art with modern
SDXL models, automatically decomposes the character into clean named layers, and
packages a complete **Live2D Cubism** project — layered PSD, physics, and the
standard parameter set — ready for auto-rigging.

Think *Stable Diffusion WebUI*, but the output is a layered, riggable VTuber
model instead of a flat image.

![status](https://img.shields.io/badge/status-beta-blueviolet)
![python](https://img.shields.io/badge/python-3.10+-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## ✨ What it does

```
 Prompt → SDXL Generate → Decompose Layers → Layered PSD + Live2D project → Rig in Cubism
```

| Stage | What happens |
|-------|--------------|
| **Generate** | Modern SDXL / Illustrious / NoobAI / Pony / Holodayo checkpoints with sampler, LoRA, clip-skip and VAE controls |
| **Decompose** | The character is split into 15+ aligned, named, transparent layers (hair front/back, head, eyes L/R, brows, mouth, body, clothing, accessories…), with occluded regions filled in |
| **Package** | A true layered **`.psd`** (plus open **`.ora`**) and a Cubism-import-ready project: `model3.json`, `physics3.json`, `cdi3.json` (pre-loaded with the standard parameter set), an idle motion and a step-by-step rigging guide |
| **Rig** | Open the PSD in **Live2D Cubism 5** and finish with its AI auto-rig |

### Highlights
- 🖥️ **Stable-Diffusion-style WebUI** — dark, fast, two-column studio with live
  WebSocket progress and ETA.
- ✂️ **Real decomposition** — region-map → SAM 2 → rembg+anatomy → geometry
  backend chain, inspired by single-image layer-decomposition research.
- 🧩 **Layered PSD / ORA export** with Live2D-friendly naming and draw order.
- 🦴 **Rig-ready Live2D scaffold** with the full standard parameter set.
- 💻 **Runs locally**, and a built-in **demo mode** exercises the entire pipeline
  with **no GPU and no model downloads**.

---

## 🚀 Quick start

### Option A — Demo mode (no GPU, instant)

Explore the whole pipeline with a procedurally-rendered character:

```bash
# Linux / macOS
./run.sh            # choose "1) Demo mode"

# or manually
python3 -m venv venv && source venv/bin/activate
pip install Flask flask-cors flask-socketio Pillow numpy psd-tools pytoshop six python-dotenv
python app.py --dummy
```

Open <http://localhost:5000> → **Studio** → *Generate Model*. You'll get a real
15-layer decomposition and downloadable PSD / Live2D package.

### Option B — Full generation (GPU recommended)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # installs PyTorch + diffusers
python app.py
```

On **Windows**, `run.bat` auto-detects your GPU and installs the correct CUDA
build of PyTorch.

### Using your own checkpoints / LoRAs
Drop files into:
```
models/checkpoints/   # .safetensors / .ckpt  (Illustrious, NoobAI, Pony, …)
models/loras/         # .safetensors LoRAs
models/vae/           # custom VAEs
```
They appear automatically in the Studio's dropdowns. Recommended 2026 anime
models: **Illustrious-XL**, **NoobAI-XL (v-pred)**, **Pony Diffusion V6**,
**Holodayo XL** (VTuber-tuned). The default, auto-downloaded model is
**Animagine XL 3.1**.

---

## 🧠 How decomposition works

The asset separator picks the best available backend:

1. **Region map** — Ka-myii's demo renderer emits a pixel-perfect sidecar.
2. **SAM 2** — Meta's Segment Anything 2 automatic masks (set
   `KAMYII_SAM2_CHECKPOINT`), classified into semantic layers.
3. **rembg + anatomy** — a foreground cut-out intersected with an anatomical
   template.
4. **Anatomy heuristic** — always-available geometric fallback.

Base layers (head, body, hair) are **hole-filled** so each exported layer is
complete even where it was occluded — the core idea behind modern single-image
decomposition (e.g. *See-through*, SIGGRAPH 2026).

> **Note on `.moc3`:** the binary runtime model can only be produced by the
> proprietary Cubism Editor, so Ka-myii prepares a complete, correctly-structured
> project right up to that final export step.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` to change the default model, resolution, sampler,
decomposition backend, etc. Every option also has a UI control.

## 🗂️ Project structure

```
app.py                     Flask entry point (+ SocketIO progress)
config.py                  Central configuration
src/core/
  image_generator.py       SDXL/SD pipeline (optional torch) + demo renderer
  demo_character.py         Procedural character + region map (demo mode)
  asset_separator.py        Layer decomposition (region-map/SAM2/rembg/geometry)
  psd_exporter.py           Layered PSD (pytoshop) + OpenRaster .ora
  model_assembler.py        Cubism-ready project packaging
  auto_physics.py           physics3.json generation
src/pipeline/assembly_line.py   Orchestration
src/api/                   REST endpoints
src/utils/model_scanner.py Local checkpoint/LoRA/VAE discovery
templates/ static/         Dark Stable-Diffusion-style WebUI
tests/test_pipeline.py     End-to-end demo-mode smoke tests
```

## 🧪 Tests

```bash
pip install pytest
python -m pytest tests/ -q          # runs entirely in demo mode, no GPU
```

## 📜 License

MIT — see [LICENSE](LICENSE).

---

*Ka-myii — from a prompt to a riggable VTuber, locally.*
