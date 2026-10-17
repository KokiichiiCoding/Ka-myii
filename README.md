# Ka-myii

**Type a character brief, get a layered, rig-ready VTuber model, all on your own PC.**

Ka-myii is a local web studio that turns a prompt into an anime character, splits it into
20+ clean, named, transparent parts with [See-through](https://github.com/shitagaki-lab/see-through)
(SIGGRAPH 2026), lets you watch it blink, talk and follow your webcam in the browser, and
packages everything for Live2D Cubism: a layered PSD, an OpenRaster file and a Cubism project
scaffold.

![Studio with a finished character and its decomposed layers](docs/images/studio.jpg)

```
brief ──► SDXL artwork ──► See-through layers ──► Live Preview ──► PSD / ORA + Cubism scaffold
          (~1–2 min)        (~10 min, 8 GB GPU)    (webcam, blink)   (finish rigging in Cubism 5)
```

## Highlights

- **Runs on an 8 GB laptop GPU.** See-through's 4-bit weights with block swapping peak at about
  4.6 GB. A character takes about 11 minutes on an RTX 4070 Laptop.
- **Real layer decomposition.** Front and back hair, face, eye whites, irises, lashes and
  brows (left and right), ears, nose, mouth, neck, clothing, accessories. Hidden areas are
  genuinely inpainted, so hair can move without leaving holes. Ka-myii also corrects
  See-through's draw-order mistakes, trims over-extended parts to what the artwork actually
  shows, drops hallucinated layers and keeps any unclaimed pixels in a `leftover` layer.
- **Live Preview.** WebGL animation with breathing, auto-blink, expressions and hair physics,
  plus optional webcam face tracking and microphone lip sync. Camera and audio never leave
  the browser.
- **A Studio that feels like a WebUI.** Your own SDXL checkpoints and LoRAs, sampler, CFG,
  seed and clip skip. Prompts longer than 77 tokens are encoded in chunks, not cut off.
  Danbooru tag autocomplete suggests 57k tags, most-used first.
- **Bring your own artwork.** Upload a character you drew or commissioned and get the same
  layers, PSD, Cubism files and Live Preview, without generating anything.
- **Background queue.** Line up several characters, reload or close the tab and come back.
  Cancel works at any stage, and unfinished jobs resume after a restart.
- **Demo mode.** The whole pipeline (layers, PSD, Cubism files, Live Preview) runs on a
  procedurally drawn character with no GPU and no downloads.

![Live Preview of a generated character](docs/images/preview.jpg)

## Requirements

| | Real generation | Demo mode |
|---|---|---|
| GPU | NVIDIA, **8 GB+ VRAM** (CUDA 12.8 driver) | none |
| RAM | 16 GB | 4 GB |
| Disk | ~25 GB (PyTorch ~5 GB, Animagine XL 4.0 ~7 GB, See-through ~9 GB) | ~1 GB |
| Python | 3.10 – 3.13 (3.12 recommended) | same |

Tested on Windows 11 with an RTX 4070 Laptop (8 GB VRAM, 16 GB RAM). On Linux, `run.sh` has
been tested on Ubuntu 24.04 (WSL) in demo mode; real generation there should work but hasn't
been tested yet. macOS has no CUDA, so it supports demo mode only.

## Quick start

### Windows

1. Install [Python 3.12](https://www.python.org/downloads/) and tick **Add python.exe to PATH**.
   Newer Pythons can stay installed; the launcher picks a supported one through the `py` launcher.
2. Download or `git clone` this repository and double-click **`run.bat`**.

On first run it creates `.venv`, installs PyTorch (with CUDA if an NVIDIA GPU is present) and
the other dependencies, offers to install See-through (recommended, ~9 GB), then asks for
**1) Studio** or **2) Demo** and opens <http://127.0.0.1:5000/generator> in your browser.
Later runs start in seconds; dependencies are reinstalled only when `requirements.txt`
changes. Keep the console window open while you work; closing it stops the server.

### Linux

```bash
./run.sh
```

On Debian/Ubuntu, install venv support first if needed: `sudo apt install python3.12-venv`
(`run.sh` tells you when it's missing).

### Manual setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
# NVIDIA on Windows: the CUDA build of PyTorch first
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/install_seethrough.py   # optional but recommended
python app.py --open                   # or: python app.py --dummy --open
```

## Using the Studio

1. **Write a brief.** Danbooru-style tags work best (`1girl, silver hair, twin braids,
   witch hat, gentle smile`). Autocomplete suggests real tags as you type; Tab or Enter accepts.
2. **Framing.** *Upper body* (the default) gives the eyes and mouth more pixels, which makes
   the live animation look much better. *Full body* shows the complete outfit.
3. **Checkpoint.** The default, **Animagine XL 4.0**, downloads automatically on first use
   (~7 GB, once). Any SDXL checkpoint you drop into `models/checkpoints/` appears in the list
   and is selected by default. Illustrious- and NoobAI-based (eps) anime models work well.
   LoRAs go in `models/loras/`.
4. **Create.** Progress shows each of the three steps with an estimate, also in the tab title.
   While a job runs, the button becomes **Add to queue**.
   To split existing art instead, choose **Use my artwork** and pick a PNG, JPG or WebP.
   One character facing forward on a plain background works best.
5. **Use the result.** Inspect the layers, open **Live Preview**, or download the **Live2D
   package**, **PSD** or **ORA**. Everything is saved under `outputs/<id>/` and listed in the
   **Gallery**, where a character can be reopened with its exact settings and seed.

### Live Preview

Breathing, blinking and gentle hair physics start immediately. **Face detail** zooms in on the
face. There are sliders for expressions, head and eyes, anchor adjustment, layer visibility and
order, transparent PNG snapshots and short recordings. **Use camera** and **Use microphone**
ask for permission only when clicked, and all processing stays in the browser (Chrome or Edge
with WebGL recommended; the camera needs `localhost` or HTTPS). Preview adjustments affect the
browser preview only; they don't rig the Cubism project.

### Finishing in Live2D Cubism

Each character's `model/<name>/` folder (the **Live2D package** download) contains:

- `<name>.psd` / `.ora`: the layered artwork, back to front, named for Cubism.
- `<name>.model3.json`, `.cdi3.json` (the standard parameter set: `ParamAngleX/Y/Z`,
  `ParamEyeLOpen`, `ParamMouthOpenY`, ...), `.physics3.json` (starter hair and cloth physics)
  and an idle motion.
- `RIGGING_GUIDE.md`, a step-by-step walkthrough for the specific character.

The JSON files validate against the schemas in Live2D's
[CubismSpecs](https://github.com/Live2D/CubismSpecs). The physics drives the standard
`ParamHairFront` / `ParamHairBack` (plus declared sway parameters for accessories and skirts).

Open the PSD in **Cubism 5**, run *Modeling → Deformer → Auto Generation of Deformer* (5.1+)
and *Auto generation of facial motion* from the Parameters palette (5.0+), then export for
runtime. The binary `.moc3` can only be produced by the Cubism Editor, so Ka-myii prepares
everything up to that step.

## How decomposition works

Ka-myii picks the best available backend:

1. **See-through** (recommended). A layer-diffusion model generates every semantic part as
   its own RGBA layer, including the parts hidden behind hair and clothes, and a depth model
   orders them. Ka-myii runs it in its own virtualenv as a separate process (so its GPU memory
   is fully released after each job), then:
   - cuts the character out with an anime segmentation model;
   - fixes draw orders that depth gets wrong for thin parts (iris under eye white, bangs under
     the face);
   - gives every visible pixel to the part the artwork really shows there, trimming
     over-extended predictions;
   - drops hallucinated layers and empty accessory specks;
   - keeps the artist's visible colours;
   - puts character pixels no layer claimed (cat ears, a witch hat) into a `leftover` layer
     so nothing goes missing.
2. **SAM 2**, if `sam2` is installed and `KAMYII_SAM2_CHECKPOINT` is set: automatic masks
   classified into semantic layers.
3. **Geometric fallback**: a rough anatomical split of the cut-out character.

Demo mode uses the procedural character's own pixel-perfect region map.

After an update, `python scripts/rebuild_layers.py <model_id>` re-applies the latest
post-processing to an existing result from its saved See-through PSD, with no GPU needed.

## Configuration

Every setting has a sensible default. To change one, copy `.env.example` to `.env`.
Common ones:

| Variable | Default | |
|---|---|---|
| `PORT` | `5000` | |
| `HOST` | `127.0.0.1` | `0.0.0.0` exposes the studio to your network, **without any login** |
| `KAMYII_MODEL_NAME` | `cagliostrolab/animagine-xl-4.0` | default Hugging Face checkpoint |
| `KAMYII_SEETHROUGH_RESOLUTION` | `1024` | `1280` gives sharper layers on 12 GB+ GPUs |
| `KAMYII_CPU_OFFLOAD` | auto below 12 GB | lower VRAM use for SDXL, slightly slower |

## Troubleshooting

- **`run.bat` says Python 3.10 to 3.13 is needed.** PyTorch's CUDA 12.8 build doesn't
  support newer Pythons yet. Install 3.12 alongside your current one; `run.bat` finds it.
- **The window closes immediately.** Open a terminal in the folder and run `run.bat` there
  to read the error.
- **"CUDA available: False" or very slow generation.** Update the NVIDIA driver, delete
  `.venv` and run `run.bat` again so it reinstalls the CUDA build of PyTorch.
- **The first character sits on "Downloading ...".** The default checkpoint is ~7 GB and
  downloads once. See-through's weights are fetched by its installer.
- **A job fails with out-of-memory, or is extremely slow.** Close other GPU-heavy apps.
  On 8 GB, keep See-through at resolution 1024. Above ~7.7 GB of VRAM use, NVIDIA drivers
  silently spill into system RAM and run ~10× slower.
- **The camera doesn't start in Live Preview.** Use Chrome or Edge on `http://127.0.0.1:5000`
  (not a LAN IP) and allow camera access.
- **Port 5000 is busy.** `python app.py --port 5001`, or set `PORT` in `.env`.

Logs are written to `logs/kamyii.log`.

## Status and known limitations

Ka-myii is an early release. What works and has been checked on real hardware: generation, See-through
decomposition, Live Preview, the queue, uploads, and the PSD/ORA/Cubism files (validated against
Live2D's published JSON schemas). Not yet confirmed by a human:

- importing a package into the Live2D Cubism Editor and running its auto-generation tools;
- webcam tracking with a physical camera (it was verified in Edge with a synthetic camera feed:
  face found, head tilt followed live);
- real (GPU) generation on Linux.

Layer quality depends on the artwork. Front-facing upper-body characters on plain backgrounds
decompose best; arms, hands and complex poses are weaker, and parts See-through cannot name end up
in `leftover` / `headwear`. Issues with example images are very welcome.

## Privacy and security

Everything runs locally; nothing is uploaded and there is no telemetry. Internet is needed
only to download dependencies and model weights. The server listens on `127.0.0.1` and
rejects requests from other sites. It has no accounts or authentication, so don't expose it
to the internet.

## Development

```bash
python -m pytest tests -q            # GPU-free; demo mode + monkeypatched See-through
node tests/preview_runtime.test.cjs  # browser-runtime checks (optional)
node tests/tagcomplete.test.cjs
```

```
app.py                       Flask entry point
config.py                    settings (.env overrides)
src/api/generation.py        REST API: /generate (queues a job), /jobs, layers, artifacts
src/pipeline/jobs.py         background job queue (one GPU job at a time, cancel, progress)
src/pipeline/assembly_line.py  generate → separate → package
src/core/image_generator.py  SDXL via diffusers (chunked prompts, LoRAs, samplers) + demo renderer
src/core/asset_separator.py  layer decomposition and See-through post-processing
src/core/seethrough.py       runs See-through in its own venv as a subprocess
src/core/psd_exporter.py     layered PSD / ORA
src/core/model_assembler.py  Cubism scaffold (model3 / cdi3 / physics3, guide)
src/core/live_preview.py     preview PSD for the browser rig
scripts/                     See-through installer and runner, rebuild_layers.py
templates/, static/          Studio, Gallery and Live Preview
```

`AGENTS.md` holds the detailed engineering notes: architecture, hard-won facts and the backlog.

## Credits

- [See-through](https://github.com/shitagaki-lab/see-through): Lin et al., SIGGRAPH 2026,
  Apache-2.0. Downloaded by `scripts/install_seethrough.py` at a pinned commit, not vendored.
- [Animagine XL 4.0](https://huggingface.co/cagliostrolab/animagine-xl-4.0): Cagliostro
  Research Lab, CreativeML Open RAIL++-M. Downloaded on first use.
- [anime-segmentation](https://github.com/SkyTNT/anime-segmentation) IS-Net weights
  (`skytnt/anime-seg`): SkyTNT, Apache-2.0, run with [ONNX Runtime](https://onnxruntime.ai) (MIT).
- [Anime2.5DRig](https://github.com/852wa/Anime2.5DRig): 852wa, MIT. It is vendored with
  local changes in `static/vendor/anime25d` (see `UPSTREAM.md`) and bundles
  [ag-psd](https://github.com/Agamnentzar/ag-psd) (MIT) and MediaPipe FaceMesh (Apache-2.0).
- Danbooru tag list from [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete),
  Dominik Reh, MIT.
- Bootstrap 5.3 and Axios (MIT); Font Awesome Free 6.4 (icons CC BY 4.0, fonts SIL OFL 1.1).
  The licenses are in `static/vendor/`.

Generated images are subject to the license of the checkpoint you use.

## License

MIT; see [LICENSE](LICENSE). Third-party components keep their own licenses (above).
