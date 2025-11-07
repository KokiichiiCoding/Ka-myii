# Ka-myii 🎨✨

**Prototype VTuber Model Generation Toolkit**

Ka-myii is a Flask-based playground for experimenting with an automated VTuber
asset pipeline.  The project stitches together a handful of Python modules that
can generate (or mock) character artwork, split that artwork into layers, and
export a very small Live2D-style package.  The codebase is intentionally
modular so you can swap in your own diffusion models, segmentation utilities, or
export logic as the project evolves.

> **Heads up**
> This repository is a prototype.  Most of the "advanced" features referenced in
> the original README were aspirational and are not implemented yet.  The
> documentation below reflects what actually ships in the code today so that you
> know exactly what to expect when you run the app.

## ✅ What currently works

- **Web UI** – Flask + Bootstrap interface with pages for generating a model,
  browsing completed runs, and opening an editor view.  You can run the server
  in dummy mode to try the interface without any ML dependencies.
- **Generation pipeline** – An `AssemblyLine` orchestrates three stages:
  1. `ImageGenerator` (Stable Diffusion via 🤗 *diffusers*, when installed) or a
     `DummyImageGenerator` that writes a coloured placeholder.  The real model is
     loaded lazily, so the app starts even if weights are not cached yet.
  2. `AssetSeparator` performs very lightweight processing.  In the default
     configuration it simply copies the base image for each requested layer and
     produces transparent placeholders for facial variants.
  3. `ModelAssembler` copies the generated layers into a `textures/` folder and
     writes simplified `model3.json` and `physics3.json` stubs so you have a
     repeatable output structure.
- **Pipeline presets & automation** – Toggle between *Lightweight*, *Standard*,
  and the new *Ultimate* preset that combines SAM-based segmentation, automatic
  accessory overlays, ControlNet-enhanced expression packs, and regenerated
  physics curves in one run.
- **Expression and accessory helpers** – Utility modules generate overlay
  graphics (eyes, brows, blush, cat ears, glasses, sparkles, etc.) on transparent
  canvases.  You can now pair them with ControlNet-powered variations straight
  from the generator UI or craft bespoke sets later in the expression editor.
- **Image editing utilities** – Pillow-based adjustments for brightness,
  contrast, saturation, sharpness, and a few filters that can be chained via the
  `/api/expression/edit` endpoint or the editor UI.
- **Downloadable results** – Every run is stored under `outputs/<model-id>/`
  with the base image, generated layers, previews, and a ZIP export endpoint.

## 🚧 What is still aspirational

The following ideas are mentioned in the codebase but are not wired up or
production ready yet:

- Automatic rigging (`AutoRigger`) only returns placeholder metadata.
- ControlNet/SAM helpers exist as modules but require additional glue code and
  heavy dependencies before they can influence results.
- Physics, animation presets, VRM export, prompt history, and model versioning
  modules are stubs and currently unused by the Flask routes.
- The generated Live2D package does **not** include a `.moc3` file.  You will
  need Live2D Cubism to create a real rig from the exported textures.

Pull requests that flesh out any of these areas are very welcome!

## 🖥️ Getting started

### 1. Clone and create a virtual environment
```bash
git clone https://github.com/yourusername/Ka-myii.git
cd Ka-myii
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies
The app imports PyTorch and 🤗 diffusers even when you run in dummy mode, so the
simplest path is to install everything from `requirements.txt`:
```bash
pip install -r requirements.txt
```
If you only plan to use dummy mode you can get away with a lighter stack
(Flask, Pillow, numpy), but you will need to edit the imports yourself.

### 3. Run the server
For a quick UI test with no GPU requirement:
```bash
python app.py --dummy
```
This route uses the dummy generator and separator so no large model downloads
are triggered.

To attempt real image generation you will need a CUDA-capable machine and the
full dependency list:
```bash
python app.py
```
The first generation will download a Stable Diffusion checkpoint via
`diffusers`.  Expect several GB of data.

The generator page exposes additional controls inspired by Gemini's creative
suite.  Pick a *Pipeline Preset* (Standard, Ultimate, or Lightweight), override
the segmentation mode, and opt into ControlNet powered expression variations or
auto-generated accessories.  Leave the comma-separated expression/accessory
fields blank to fall back to the preset bundles (`config.DEFAULT_*`).

### 4. (Windows) Helper script
`run.bat` automates the virtual environment setup and dependency installation
before asking whether to start the app in dummy or production mode.  It relies on
`python` being available in `PATH` and, for the production option, a working
CUDA toolchain.

Once the server is running open [http://localhost:5000](http://localhost:5000)
for the landing page.

## 📂 Project layout
```
Ka-myii/
├── app.py                 # Flask entry point and blueprint registration
├── config.py              # Centralised configuration
├── requirements.txt
├── src/
│   ├── api/               # Flask blueprints for generation/expression/model info
│   ├── core/              # Generation, separation, assembly, and helper modules
│   ├── models/            # Dataclasses describing requests and outputs
│   ├── pipeline/          # AssemblyLine orchestrator
│   └── utils/             # Logging, progress, history stubs
├── templates/             # Bootstrap-powered UI
├── static/                # Front-end assets
└── outputs/               # Created at runtime to hold results
```

## 🌐 API overview
All endpoints are prefixed with `/api`:

- `POST /api/generation/generate` – Run the pipeline and return metadata for the
  new model.
- `GET /api/generation/list` – Enumerate saved runs from the `outputs/` folder.
- `GET /api/generation/preview/<model_id>` – Fetch the preview PNG.
- `POST /api/generation/preload-model` – Manually trigger Stable Diffusion model
  loading.
- `POST /api/expression/generate` – Draw expression layers (eyes, brows, blush,
  etc.) for a stored model image.
- `POST /api/expression/accessories/generate` – Generate overlay accessories
  using presets or a custom list.
- `POST /api/expression/edit` – Apply Pillow adjustments to a stored image.
- `GET /api/models/info` / `GET /api/models/stats` – Basic metadata about the
  environment and generated runs.

Every route returns JSON responses and surfaces errors with appropriate HTTP
status codes.

## ⚠️ Known limitations

- Asset separation is **rule-based** and will not magically produce clean Live2D
  parts.  Treat the output as a starting point for manual clean-up.
- The generated Live2D files are structural placeholders—there is no automatic
  `.moc3` mesh or physics tuning.
- Expression and accessory assets are simplistic vector drawings intended for
  prototyping.  They do not analyse the source image.
- Running the full pipeline requires a large GPU memory footprint (≈10 GB) and a
  reliable internet connection for model downloads.

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-change`).
3. Make your updates and add tests where practical.
4. Run the formatter/linter if you have them installed (`black`, `flake8`).
5. Submit a pull request describing what changed and how to exercise it.

## 📄 License

Released under the MIT License.  See [`LICENSE`](LICENSE) for full text.

---

Made with ❤️ for VTuber pipeline experiments.
