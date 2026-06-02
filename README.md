# Ka-myii 🎨✨

### **Automated VTuber Model Studio**

> Turn a text prompt into a **rig-ready Live2D character** in minutes

Ka-myii is a local, self-hosted web app that generates high-quality anime art with modern
**SDXL models**, automatically decomposes characters into **15+ named layers**, and
packages a complete **Live2D Cubism project** — layered PSD, physics, parameters and all —
ready for Cubism 5's AI auto-rig.

**Think *Stable Diffusion WebUI*, but the output is a riggable VTuber model instead of a flat image.**

![status](https://img.shields.io/badge/status-beta-blueviolet)
![python](https://img.shields.io/badge/python-3.10+-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![GPU](https://img.shields.io/badge/GPU-optional-orange)

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

### Windows Users 🪟

**Ka-myii includes a smart `run.bat` script that handles everything automatically!**

#### Quick Start (3 clicks):
1. **Download** this repository (Code → Download ZIP) and extract it
2. **Double-click `run.bat`** (in the Ka-myii folder)
3. **Choose your mode** when prompted:
   - `1` = Production mode (GPU/CUDA if available)
   - `2` = Demo mode (no GPU needed, instant start)

> **⚠️ If `run.bat` fails or closes immediately**: Try `run-simple.bat` instead (simpler version with better error messages)

The script will:
- ✅ Check if Python is installed (install from [python.org](https://www.python.org/) if not)
- ✅ Create a virtual environment automatically
- ✅ Detect your NVIDIA GPU and install CUDA-enabled PyTorch
- ✅ Install all dependencies
- ✅ Start the web server

Then open <http://localhost:5000> in your browser! 🎨

#### Demo Mode (No GPU Required) 🎬
Perfect for trying Ka-myii without downloading AI models:
- Double-click `run.bat` → Choose option `2`
- Opens instantly, uses procedural rendering
- Full 15-layer decomposition + PSD/Live2D export
- Test the entire pipeline with zero GPU/VRAM

#### Production Mode (GPU Recommended) ⚡
For real SDXL anime generation:
- Double-click `run.bat` → Choose option `1`
- Auto-detects CUDA and installs GPU-accelerated PyTorch
- First run downloads Animagine XL (~7GB) automatically
- Recommended: NVIDIA GPU with 8GB+ VRAM

#### Manual Setup (PowerShell/Command Prompt)
If you prefer manual control:

```powershell
# PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py

# Demo mode (no GPU)
python app.py --dummy
```

```cmd
REM Command Prompt
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python app.py

REM Demo mode (no GPU)
python app.py --dummy
```

### Linux / macOS Users 🐧🍎

```bash
# Quick start with interactive menu
./run.sh            # choose "1) Demo mode" or "2) Full/GPU"

# Or manually
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py

# Demo mode (no GPU needed)
python app.py --dummy
```

---

### After Launch

Once the server starts, you'll see:
```
======================================================================
Ka-myii: Automated VTuber Model Generation System
======================================================================
Starting server on 0.0.0.0:5000
```

Open your browser to: **<http://localhost:5000>**

Click **"Open the Studio"** → Enter a prompt → Generate! 🎨✨

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

## 🪟 Windows Troubleshooting

### Common Issues & Solutions

#### `run.bat` fails or closes immediately
**Symptoms**: Window flashes and closes, or stops after "Upgrading pip..."

**Quick Fix**: Use `run-simple.bat` instead (simpler version with better error reporting)

**Debug Steps**:
1. **Don't double-click** - instead, open Command Prompt in the folder:
   - Hold `Shift` + Right-click in the Ka-myii folder
   - Choose "Open Command window here" or "Open PowerShell window here"
   - Type: `run.bat` and press Enter
   - This keeps the window open so you can see the error

2. **Common causes**:
   - **Antivirus blocking**: Windows Defender may block pip installs
     - Solution: Add Ka-myii folder to exclusions
   - **Permissions**: Script can't write to folder
     - Solution: Right-click run.bat → Run as Administrator
   - **Network proxy**: Pip can't download packages
     - Solution: Check your internet connection / proxy settings
   - **Corrupted pip**: Python installation is broken
     - Solution: `python -m pip install --upgrade pip --user`

3. **Manual install** (if batch file won't work):
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install Flask flask-cors flask-socketio Pillow numpy psd-tools pytoshop six python-dotenv
python app.py --dummy
```

#### "Python is not recognized" or "python: command not found"
**Solution**: Python isn't in your PATH
1. Download Python from [python.org](https://www.python.org/) (3.10 or newer)
2. **Important**: Check "Add Python to PATH" during installation
3. Restart your terminal/command prompt after installing

#### "Scripts execution is disabled on this system" (PowerShell)
**Solution**: PowerShell execution policy is restricted
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Or use Command Prompt (`cmd.exe`) instead!

#### "CUDA out of memory" errors
**Solution**: Your GPU doesn't have enough VRAM
1. Close other GPU-using apps (Chrome, games, etc.)
2. Lower resolution: Try 512×768 or 640×896 in the Studio
3. Set `KAMYII_CPU_OFFLOAD=true` in `.env` (slower but uses less VRAM)
4. Use demo mode: `python app.py --dummy`

#### Windows Defender / Antivirus blocking
**Solution**: Add exception for Ka-myii folder
- Windows Defender → Virus & threat protection → Manage settings → Add exclusion
- Add the entire `Ka-myii` folder

#### Firewall blocks the server
**Solution**: Allow Python through Windows Firewall
- When Windows asks "Allow Python to communicate?", click **Allow access**
- Or manually: Windows Firewall → Allow an app → Python

#### Very slow generation (no CUDA detected)
**Solution**: PyTorch can't see your GPU
1. Update NVIDIA drivers: [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Restart PC after driver update
3. Re-run `run.bat` to reinstall PyTorch with CUDA
4. Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`

#### Port 5000 already in use
**Solution**: Another app is using port 5000
```cmd
# Use a different port
python app.py --port 5001

# Or find what's using port 5000
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```

#### Missing DLL errors (VCRUNTIME, msvcp)
**Solution**: Install Visual C++ Redistributable
- Download from [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- Install both x64 and x86 versions

---

## 🚀 Hosting & Testing Options

### For Local / Personal Use
```bash
python app.py                      # Local only (localhost:5000)
python app.py --host 0.0.0.0       # Allow LAN access (same WiFi)
```

### For Quick Sharing / Remote Testing
**ngrok** (free tier available):
1. Start Ka-myii: `python app.py`
2. In another terminal: `ngrok http 5000`
3. Share the `https://xxxx.ngrok.io` URL anywhere!

### For Production / Cloud Deployment

Ka-myii can be deployed to:
- **Hugging Face Spaces** (free GPU tier, best for demos)
- **Railway / Render** (~$5-10/month, always-on)
- **AWS / GCP / Azure** (full control, production)
- **RunPod / Vast.ai** (cheap GPU hourly rates)
- **Docker** (coming soon)

**📖 Complete deployment guide**: See **[DEPLOYMENT.md](DEPLOYMENT.md)** for step-by-step instructions, security checklists, and production configurations

---

## 🧪 Tests

```bash
pip install pytest
python -m pytest tests/ -q          # runs entirely in demo mode, no GPU
```

## 📜 License

MIT — see [LICENSE](LICENSE).

---

*Ka-myii — from a prompt to a riggable VTuber, locally.*
