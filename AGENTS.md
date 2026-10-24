# Ka-myii — agent handoff & working guide

You are continuing development of **Ka-myii**, a local, self-hosted web app that turns a text
prompt into a **rig-ready VTuber model**: SDXL anime image → decomposed into 20+ named,
inpainted layers → layered PSD/ORA + Live2D Cubism project scaffold. The owner wants it to be a
genuinely better application and is fine with using any open-source library, model or asset
(respect licenses, credit them in the README).

Read this whole file before changing anything. Keep it up to date when you finish a milestone.

---

## 1. Product decisions already made by the owner (don't re-litigate)

- **Self-contained**: Ka-myii installs and runs its own PyTorch stack. It must NOT depend on
  ComfyUI or any other app being installed.
- **End goal = Cubism handoff + live preview**: keep the best possible PSD / Live2D Cubism project
  export, AND add an in-browser preview where the character is animated (blink, mouth, head tilt,
  hair sway) driven by the user's webcam — so they see a working VTuber before opening Cubism.
- Target hardware is the owner's laptop: **RTX 4070 Laptop, 8 GB VRAM, 16 GB RAM, Windows 11**.
  Everything must work on 8 GB VRAM. RAM is tight (the owner keeps browsers/Discord/Steam open).

## 2. Repository state — READ FIRST

- Path: `C:\Users\Kokii\Ka-myii` — git clone of `github.com/KokiichiiCoding/Ka-myii`.
- Branches: **`main`** (created 2026-10-24 by the owner's choice from the development branch at
  `7bdb230`; the owner switches GitHub's default to it in Settings) and
  `claude/automated-vtuber-model-011CUg6gEfYhoWvMuuXdF2hJ` (the development branch, kept in step
  with main). The old default `Ka-myii` is a stale ancestor; leave it alone.
- **M1–M5 are committed** (2026-10-17); the 2026-10-18 preview motion polish and
  manual blink editor, and the 2026-10-23 motion/companion, nod, neck/tilt, shoulders/collar/hat/extend, torso/mouth/full-body UI, seamless-extend/toggles, guided-extension/fidelity and stance/checks/visemes/variants passes below are uncommitted. The owner asked to make Ka-myii a good initial
  product to push to GitHub; M1–M3 were committed as one baseline (`c113da1`), M4 work
  (section 9) and M5 (section 10) as separate commits on top. Commit or push only when the owner asks.
  **Never run `git checkout .`, `git reset --hard`, `git stash` or `git clean` on this tree**
  (`outputs/` holds the owner's characters).
- **The repo is public.** The local git config's email is the owner's personal address; commit as
  `git -c user.name=KokiichiiCoding -c user.email=130875715+KokiichiiCoding@users.noreply.github.com commit …`
  (the GitHub noreply identity earlier commits use) so pushes don't publish it.
- Never commit model weights, `outputs/`, `engines/`, venvs (all gitignored — keep it that way).

## 3. Environment & commands (Windows, PowerShell)

| What | Command / path |
|---|---|
| App venv | `.venv\` (Python 3.12; torch 2.8.0+cu128, diffusers 0.37.0, transformers 5.0.0, accelerate 1.13.0, peft, Flask, psd-tools, pytoshop). `run.bat` now prefers an existing `.venv\`, then `venv\`, and creates `.venv\` otherwise. |
| See-through engine | `engines\see-through\` (git clone, pinned commit `df019de`), its own venv `engines\see-through\.venv\` (torch 2.8.0+cu128, bitsandbytes 0.49.2, diffusers 0.37.0, transformers 5.0.0) |
| Model weights | See-through NF4 weights in the HF cache (`%USERPROFILE%\.cache\huggingface\hub`, repos `24yearsold/seethroughv0.0.2_layerdiff3d_nf4`, `24yearsold/seethroughv0.0.1_marigold_nf4`). SDXL checkpoint `models\checkpoints\novaAnimeXL_ilV140.safetensors` (Illustrious-based; a **hard link** to the owner's ComfyUI copy — never delete or modify it in place). |
| Tests (no GPU, no torch needed) | `.\.venv\Scripts\python.exe -m pytest tests --ignore=tests/e2e -q` → 98 must pass; on sandbox permission conflicts add `-p no:cacheprovider --basetemp <fresh dir>`. Plus `node tests/preview_runtime.test.cjs` and `node tests/tagcomplete.test.cjs`. CI (`.github/workflows/tests.yml`) runs the same on Windows + Ubuntu without torch. `tests/conftest.py` forces `seethrough.available() == False` so no test can start the real engine. |
| Re-run layer post-processing on old results (no GPU) | `.\.venv\Scripts\python.exe scripts\rebuild_layers.py <model_id> [...]` — re-separates from the saved See-through PSD, regenerates PSD/ORA/Cubism/preview/metadata |
| Demo mode (no GPU) | `.\.venv\Scripts\python.exe app.py --dummy --host 127.0.0.1 --port 5055` |
| Real mode | `.\.venv\Scripts\python.exe app.py --port 5055` (add `--open` to launch the browser). Production + 127.0.0.1 are now the defaults; `FLASK_ENV=development` brings back the Werkzeug reloader, which **restarts the server (killing a running job) whenever you edit a file**. Templates are cached in production: restart after editing them. |
| Run See-through alone | from `engines\see-through`: `.\.venv\Scripts\python.exe -u ..\..\scripts\seethrough_runner.py --srcp <img> --save_dir <dir> --resolution 1024 --tblr_split` |
| GPU check | `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv` |

Real-mode timings on the owner's laptop (after the M3 RAM fix): model load ~65 s cold, SDXL
832×1216 28 steps ~1–2 min, See-through ~10 min, total ~11 min (653 s measured).

**A ready-made real result exists for development:** `outputs\dd6e1715-5d8d-462a-a92d-279404ed02e0\`
(pink-haired idol, 21 See-through layers). Contains `base_image.png`, `assets\NN_<layer>.png`
(full-canvas, aligned), `assets\seethrough\base_image.psd` (See-through's own PSD, 1024² padded
square) and `model\model_8579bb70\` (Ka-myii's PSD/ORA/Cubism files). Use it to build M2 without
re-generating. Other `outputs\` folders are demo-mode results.

## 4. Architecture

```
app.py                         Flask (no CORS, no SocketIO); Host-header check when bound to loopback; --open, --dummy
config.py                      all settings; loads .env (python-dotenv)
src/api/generation.py          REST: /generate (validated brief → job, 202), /upload (user artwork → job), /jobs, /jobs/<id>, /jobs/<id>/cancel,
                               /options, /layers, /layer (?crop=1), /artifact, /preview … — every URL model id goes through model_dir_for()
src/pipeline/jobs.py           JobQueue: one daemon worker runs jobs in order; stage-weighted progress; cancel;
                               real mode persists unfinished jobs to outputs/_queue.json
src/pipeline/assembly_line.py  orchestration: generate or import upload → (unload SDXL) → separate → assemble → preview;
                               stage + cancel checks; a failure with artwork is saved as status failed
src/core/image_generator.py    diffusers SDXL; chunked long-prompt encoding; auto model-CPU-offload when VRAM < 12 GB; DummyImageGenerator for demo
src/core/asset_separator.py    layer decomposition. Priority: demo region-map → See-through → SAM2 → geometric fallback.
                               See-through path: draw-order priors → speck / degenerate-face-detail / hallucination filters →
                               per-pixel ownership + source colours → leftover split at the chin (above → headwear)
scripts/rebuild_layers.py      re-run post-processing + exports for existing results from their saved See-through PSD
static/js/tagcomplete.js       Danbooru tag autocomplete (data: static/vendor/tags, MIT, a1111-sd-webui-tagcomplete)
src/core/seethrough.py         spawns the See-through runner as a subprocess; parses its stdout into progress
scripts/seethrough_runner.py   runs inside the engine venv (cwd = engines/see-through); NF4 + block swap
scripts/install_seethrough.py  installer for other users (download pinned zip, venv, deps, weights); run.bat offers it
src/core/psd_exporter.py       Ka-myii's PSD (pytoshop) / ORA / flat composite from assets
src/core/model_assembler.py    Cubism scaffold: model3/physics3/cdi3 json, idle motion, RIGGING_GUIDE.md
src/core/auto_physics.py       physics3.json from layer names (keywords: hair+front/back, cloth…)
templates/, static/            "Studio" UI (generator.js: form, job polling, queue panel, layer grid)
tests/test_pipeline.py         demo-mode end-to-end + See-through mapping/filter/order/cutout tests (GPU-free, monkeypatched)
tests/test_jobs.py             job queue order/cancel, assembly-line cancel cleanup, See-through progress + kill
```

Layer naming follows See-through's tags (keep it — the live-preview tools expect it):
`back hair, front hair, face, eyewhite-l/-r, irides-l/-r, eyelash-l/-r, eyebrow-l/-r, ears-l/-r,
nose, mouth, neck, neckwear, topwear, handwear-l/-r, bottomwear, legwear, footwear, headwear,
earwear, eyewear, tail, wings, objects`. Assets are full-canvas RGBA PNGs aligned to `base_image.png`,
ordered back→front by `metadata.order` (filename prefix `NN_`).

## 5. Hard-won facts — do not regress these

1. **See-through's stock 8 GB scripts crash on this Windows setup.** diffusers *group offload* of
   bitsandbytes NF4 weights → `illegal memory access` / `CUBLAS_STATUS_NOT_INITIALIZED` /
   `CUBLAS_STATUS_ALLOC_FAILED`. Don't use `--group_offload` or `inference_psd_quantized.py` defaults.
2. **NVIDIA sysmem fallback**: overflowing 8 GB does not raise OOM — it silently spills into shared
   system RAM and runs ~10× slower. Watch `nvidia-smi`; > ~7.7 GB used means you are spilling.
3. VRAM is dominated by **activations** (all 13 body tags run as one batch): ~8 GB at 1280, ~3.5 GB
   at 1024. Default `KAMYII_SEETHROUGH_RESOLUTION=1024`.
4. The working recipe (`scripts/seethrough_runner.py`): build the NF4 pipeline with no offload →
   it caches tag embeddings and drops the text encoders → re-class it as the repo's
   `KDiffusionStableDiffusionXLPipelineBlockSwap` (skip its `__init__`) → cast the UNet's floating
   params/buffers (fp16 in the NF4 checkpoint) and the cached prompt embeds (fp32) to **bf16** →
   `enable_blockswap`. Peak 4.6 GB.
5. The app must call `image_generator.unload_model()` (with `gc.collect()`) before See-through runs —
   both cannot share 8 GB. The subprocess design guarantees See-through's VRAM is freed on exit.
6. Pillow 12: `Image.fromarray(arr)` shares the numpy buffer, so `ImageDraw.floodfill` edits are not
   visible via `np.asarray` — `.copy()` first (fixed in `_fill_holes`, covered by a test).
7. See-through occasionally hallucinates layers (seen: grey "headwear" blobs). `_separate_seethrough`
   drops a layer if > 50 % lies outside the character silhouette, or if it covers > 0.2 % of the image
   and its visible pixels differ from the source by mean > 60. Tuned on two images only — it catches
   gross cases; a ribbon/headwear mix (error ≈ 41) still slips through. Don't lower 60 blindly:
   real thin parts (lashes, brows, glasses) score 45–110 due to resampling.
8. Frontend progress: `currentTaskId` must be reset before each POST (fixed) or updates get filtered.
9. Cubism FREE limits: one texture ≤ 2048 px, ≤ 100 ArtMeshes. Real Cubism AI tools:
   *Modeling → Deformer → Auto Generation of Deformer* (5.1+), *Parameters palette menu → Auto
   generation of facial motion* (5.0+). There is no "Auto-Rig (AI)" menu.
10. **PyTorch's Windows CPU allocator is mimalloc and only returns freed memory to the OS during
   later allocations.** After `unload_model()` every tensor/module was gone (verified with `gc`),
   yet the idle app kept ~7.5 GB private — See-through then ran swapping, with 450 MB RAM free.
   `app.py` sets `MIMALLOC_PURGE_DELAY=0` before torch loads: private memory after unload went
   11.3 GB → 3.8 GB. Keep it first in `app.py`; any other torch entry point needs it too.
11. **diffusers `clip_skip` is not A1111's.** SDXL uses `hidden_states[-(clip_skip + 2)]` for the
   positive prompt only (negative always uses −2), so the UI's "2" meant layer −4 vs −2. Ka-myii
   now encodes SDXL prompts itself (`_sdxl_prompt_embeds`): A1111 meaning, both prompts, and
   75-token chunks so long briefs aren't cut at 77. Verified bit-exact (max diff 0.0) against
   `pipe.encode_prompt` for short prompts. SD1.5 still uses diffusers (`clip_skip - 1`), truncates.
12. **See-through's layer order (predicted depth) is unreliable for thin, coplanar parts.** Seen:
   `irides-l` under `eyewhite-l`, `neck` over `topwear`. The colour-error filter then deleted
   the real eye white and neck (errors 72/76). `ORDER_PRIORS` in `asset_separator.py` fixes
   such pairs before filtering; everything else keeps See-through's order. Extend the list
   when a new mis-order is observed — don't loosen the colour threshold.
13. See-through emits few-pixel accessory specks (`headwear`/`eyewear`/`earwear`, 4–6 px at
   the same spot) when there is no accessory; they're dropped (< 0.05 % of the image, accessory
   tags only — facial parts are legitimately tiny). It can also assign real pixels to nothing
   (seen: cat ears); those go to a `leftover` layer (live preview maps it to the rig's `objects`),
   now split at the chin — see fact 18.
14. **The character cut-out matters more than it looks** (silhouette for the hallucination check,
   the `leftover` layer, background clearing). The old median-colour key failed on upper-body
   crops (99.8 % "character"). Now: SkyTNT's anime-segmentation IS-Net via onnxruntime
   (`_anime_seg_mask`, ~2 s CPU, weights read only from the local HF cache — the installer
   fetches them). Fallback colour key: most common border colour, border-connected region,
   flooded from top/sides only (a white shirt touching the bottom edge was being keyed out),
   and it still fails on gradient/vignette backgrounds — so per-pixel background clearing only
   runs when the model produced the mask (`cutout_backend == "anime-seg"`). Tests force the
   colour key via `tests/conftest.py`. rembg is not installed (its deps are ~200 MB).
15. **Per-pixel ownership** (`_resolve_ownership`): See-through over-extends parts (bangs across
   the cheeks). After whole-layer filtering, each pixel goes to the part whose predicted colour
   best matches the source (topmost within a margin of 30, so depth order wins when unclear);
   parts above it are cut there, then the owner gets the source colour. On four real results this
   lowered composite-vs-source error with identical masks (e.g. 3.50 → 3.40, 7.38 → 7.15), and it
   removed bangs that had been baked into face layers / skin baked into hair layers.
16. Cancel latency: `seethrough._follow` reads output on a pump thread and re-calls `progress`
   every second, so a cancel lands even while the runner is silently loading weights.
   `atexit` kills in-flight runners (Windows does not kill children with the parent).
17. **See-through can emit a degenerate facial-detail layer.** One real run's `nose` was a 60 %
   copy of the upper face (eyes, bangs, cheeks); it matches the source, so the colour filter kept
   it and ownership then gave it the visible cheeks. Across five runs every real nose / mouth /
   brow / lash / iris / eye white was ≤ 3 % of the face's area, so such layers are dropped above
   20 % (`MAX_FACE_DETAIL_SHARE`).
18. **The browser rig moves each layer with either the head or the body** (an unknown/`objects`
   layer goes by its centroid vs the face bottom). A single `leftover` holding a halo above the
   head and chest skin below therefore tore somewhere. `_place_leftover` splits it at the chin:
   above → `headwear` (merged into a real headwear layer — lossless, uncovered pixels never
   overlap a kept part's opaque ones), below stays `leftover` above the back hair.
19. **Fresh installs resolve the newest ML packages** (diffusers 0.41 / transformers 5.20 on
   2026-10-17 vs the verified 0.37 / 5.0). `requirements.txt` pins diffusers, transformers,
   accelerate, peft and onnxruntime to the verified versions; bump only after a real GPU run.
   PyTorch 2.8 cu128 has no Python 3.14 wheels, so `run.bat` picks 3.10–3.13 via the `py`
   launcher (`run.sh` likewise).
20. **Local-server security.** Blanket CORS + an unvalidated model id meant any website could
   `DELETE /api/generation/delete/..` (rmtree of the project). Now: no CORS, no SocketIO,
   `model_dir_for()` for every URL id, `/generate` accepts only listed checkpoints/LoRAs
   (it loaded any path on disk), bounded numbers, and a Host-header check against DNS
   rebinding when bound to loopback. Uploads are re-encoded by PIL before anything uses them.

21. **Cubism files are checked against Live2D's official JSON schemas** (embedded in the
   markdown of github.com/Live2D/CubismSpecs, `FileFormats/*.json.md`; that repo has no license,
   so they are not vendored). model3/cdi3/motion3 passed; physics3 lacked required Meta fields
   and drove undeclared, invalid ids — fixed. `test_cubism_files_follow_the_specs_…` checks the
   rules without the schemas; re-validate with them after changing the assembler.

## 6. Known limitations / backlog

- Full-body characters still have small facial parts. Studio now defaults to upper-body framing;
  full-body remains selectable. Preview has a Face detail zoom; zoom does not add source pixels.
- `handwear-l/-r` split sometimes fails (both arms + a skirt frill landed in `handwear-l`).
- Real mode persists unfinished jobs to `outputs/_queue.json` and restores them (same ids,
  queued; an interrupted job restarts from scratch, at most twice). Demo mode keeps no state.
  A crashed run's half-written output folder (no metadata.json) is not cleaned up automatically.
- Output has never been opened in Live2D Cubism by a human — needs a real import check.
- Webcam tracking was verified with a **synthetic camera** (2026-10-17): headless Edge via
  playwright-core with `--use-fake-device-for-media-stream --use-file-for-fake-video-capture=<y4m>`
  (a public-domain portrait rotated ±0.3 rad, 4 s period). Camera on, offline FaceMesh found the
  face, tracked roll oscillated −0.47…+0.35 in phase with the video, the rig's head followed, the
  `headwear` hat moved with the head, 51 fps, no console errors. Read state via
  `window.Anime25D.state()` (`camLive`, `cam.az`, `pose.angleX`). The same fake device's beeping
  microphone drove lip sync (mouth 0 until the mic started, then 0.4–0.99 with the beeps).
  A physical webcam/microphone is untried.
- `run.sh` was run on Ubuntu 24.04 under WSL (2026-10-17): venv check message, torch 2.8 + pinned
  requirements, demo server, a demo job with package + preview PSD downloads. Real generation /
  See-through on Linux are untested. macOS can only use demo mode (no CUDA).
- `run.bat`'s prompts: `fc` (the requirements check) was swallowing redirected stdin, so piped
  answers never reached `set /p`; fixed with `<nul`, and a piped "2" then started demo mode.
  Prompts typed in a real console have not been exercised by a human yet.
- The old expression editor / accessory generator / image editor were removed in M4 (they pasted
  clip-art shapes at guessed positions); Live Preview covers expressions.

## 7. M2 — live preview + character workshop (complete)

- Vendor: `static/vendor/anime25d/`, Anime2.5DRig MIT, pinned to
  `7ddbd9943ea3152561b3dc8348fd752c850f3e95`. Read `UPSTREAM.md`; licenses are included.
  Actual source is classic scripts + a PSD parsing worker (not ES modules).
- Upstream `?model=` only auto-loads in OBS mode and accepts relative files. Ka-myii instead
  injects a server-generated same-origin URL into `window.KamyiiPreview`; the small adapter
  calls upstream `loadModel` in the normal editor. No arbitrary URL query input was enabled.
- `/preview/<model_id>` opens the stage. Studio and Gallery link to it.
  `src/core/live_preview.py` merges paired PNGs, maps old demo layer names, preserves original
  dimensions/offsets/order, and caches an atomic PSD in `outputs/<id>/live_preview/`.
  `/api/generation/artifact/<id>/preview_psd` serves it; source PNGs/Cubism files are untouched.
- **Source inspection changed the expression plan:** v2 synthesizes BOTH closed eyes and closed
  mouth from bundled PSD/generic artwork, tinted and anchored to the character. SDXL inpainting
  is not required. Preview runs without torch. Artist-provided close-expression layers work too.
- Local adapter: English primary controls, calmer hair/body motion, Face detail framing,
  a 60fps cap, no CDN fallback for FaceMesh, no OBS relay probes/polling in Ka-myii.
  Upstream noise filtering removed tiny facial parts; local threshold is 4 pixels instead of 40.
- Offline UI libraries are in `static/vendor/ui/` with licenses. No Google font/CDN dependency
  remains on Studio/Gallery. Preview requires WebGL; camera requires WebAssembly SIMD and
  localhost/HTTPS. Camera/mic are off by default, input stays local, pending startup is cancellable.
- Browser settings/anchor changes belong to the preview, not the Cubism scaffold. No `.moc3`
  export or automatic production Cubism rig is claimed.
- Studio/Gallery/home use a charcoal, warm-paper/copper workshop theme. Gallery preserves
  export, expression-editor and confirmed-delete actions. Saved models reopen in Studio.
- `GenerationRequest.framing` supports `full_body` (API/backward-compatible default) and
  `upper_body` (Studio default). Upper body removes full-body/reference-sheet preset tags.
  Requests are saved in metadata; random seeds are resolved once and retained for reuse.
- Artist PSD export now passes explicit `(width, height)` to pytoshop. Omitting `size`
  crops the canvas AND shifts layer offsets; its docstring says the opposite axis order to its
  implementation. Regression test covers margins/offsets. Preview export uses psd-tools>=1.25;
  a negative PSD layer count is needed to declare merged transparency for new RGBA documents.
- Cubism-package download now targets `model/<name>/` instead of zipping every workfile.
  Archives stream through an anonymous temporary file to avoid Windows locks/temp leaks.

Verification (2026-10-15):

- Original 6 pytest tests passed before M2. Current 15 GPU-free pytest tests pass (9 preview/output + 6 pipeline).
- `node tests/preview_runtime.test.cjs` passes: alias mapping, rig anchors, synthesized close
  expressions, offline-only devices, no camera, denied mic and late permission stream cleanup.
- In-app browser/Playwright: original idol renders 23 animated parts, both eyes/irises/lashes/
  brows/mouth recognized, repeated actual blink cycles observed, wink visibly rendered,
  Face detail works, no console warnings/errors. A fresh demo result
  `8e076563-6d22-4263-950f-294c9aa8b091` renders 16 animated parts through Studio's preview button.
- Gallery search/card links checked in browser, including a phone-width layout without overflow.
- In-app camera permission stayed pending; cancellation verified. **Physical webcam tracking
  has not been verified**; check Chrome/Edge with permission granted on the owner's camera.
- Real upper-body run **completed**: `b6f6fe2f-96c6-46c3-af08-05a73e80b432`, model
  `model_008ccaf9`, seed 271828, novaAnimeXL, 832×1216, 28 DPM++ 2M Karras steps.
  GPU pipeline took **1195.9 s (19m56s)** in this session. Highest observed SDXL memory
  sample: **5949 MiB**; decomposition monitor maximum: **4564 MiB** (10-second samples,
  not a guaranteed instantaneous peak). No spill observed. GPU returned to 10 MiB idle.
- Real run exposed an erroneous mouth rejection: chroma foreground detection had an internal
  hole over pale skin. Silhouette validation now fills **interior** holes before dilation;
  an exterior headwear blob is still rejected in the regression test.
- Visible retained layer pixels now use original source RGB, while hidden inpainting stays
  untouched. This fixes a grey neck/skin colour shift seen in the real result. Rejected layers
  no longer add occlusion to later validation. Covered by an overlapping-layer regression test.
- Rebuilt this new result's assets and exports **from its existing See-through PSD, with no GPU
  rerun**. It has 17 artist layers + the helper cutout; preview renders 19 animated parts and
  detects both eyes, irises, lashes, brows and mouth. Artist PSD canvas checked at 832×1216.
- Final browser check on the rebuilt real output: 19 animated parts, 6 automatic blink
  cycles observed, demo-speech mouth openness reached 0.84, every core facial diagnostic
  passed, and no console warnings/errors. Camera/mic left off; demo speech left off.
  The new home page showcases the latest real output.
- Browser screenshots live in ignored `outputs/m2-review/`. No model weights were modified.
  A real-mode server is running at http://127.0.0.1:5055 with the reloader disabled.

(The 77-token truncation noted here during M2 is fixed in M3, see fact 11.)

## 8. M3 — background jobs, prompt quality, decomposition fixes (complete, 2026-10-16)

- **Background job queue** (`src/pipeline/jobs.py`): `/generate` returns 202 + job; one worker
  thread runs jobs in order. `GET /jobs`, `GET /jobs/<id>` (state, stage, overall progress, per-step
  ETA, queue position), `POST /jobs/<id>/cancel`. Overall progress is stage-weighted (image 0–15 %,
  separate 15–97 %, assemble 97–100 %) from measured timings and never goes backwards; See-through's
  body+head passes are counted as two tqdm bars. Cancel works at every stage: diffusion step
  callback, See-through runner killed (even while silent), stage boundaries; the half-built output
  folder is deleted. `ProgressManager` keeps cancellation/stage per task across per-stage trackers.
- **Studio**: polls the job (single source of truth; the socket is no longer used by Studio),
  re-attaches after reload, Cancel button, "Step n of 3 · …" label, progress in the tab title,
  and a queue: while a job runs the button becomes "Add to queue"; a Queue list shows other jobs
  with Watch / Cancel.
- **Prompts**: chunked SDXL encoding (no 77-token cut), A1111 clip-skip semantics (fact 11).
  Danbooru tag autocomplete on prompt/negative (`static/js/tagcomplete.js`, 57k tags with ≥ 100
  posts, aliases, most-used first; ↑/↓, Enter/Tab, Esc).
- **Decomposition quality** (facts 12–16): draw-order priors, accessory speck filter, per-pixel
  ownership, `leftover` layer, anime-segmentation cut-out (new app dependency `onnxruntime`;
  installed 1.30.0 — 1.31.0 was under two weeks old) with background clearing.
  `scripts/rebuild_layers.py` upgrades older results without a GPU; all four real results in
  `outputs/` were rebuilt with it (previous layers backed up outside the repo, not needed).
- **RAM**: `MIMALLOC_PURGE_DELAY=0` (fact 10).
- Fixed `ProgressTracker.complete(status=…)` TypeError (cancel path called an argument it lacked).

Verification (2026-10-16, owner's laptop):

- 24 pytest + 2 node tests pass. Demo-mode browser checks: generate, reload re-attaches, cancel,
  queue add / list / cancel / watch, tag autocomplete keyboard flow, no console errors.
- Real run A (`8feed57f-890e-474b-9da3-b606196d0d31`, seed 424242, cat-ear hoodie girl, long
  prompt → "2 chunks of 75 tokens"): completed through the Studio in 1011 s while a pip install
  ran alongside and RAM was starved (pre-mimalloc fix). Job B queued behind it started
  automatically; cancelled 60 s into See-through: runner killed, output folder removed, VRAM
  back to 169 MiB. A was then rebuilt with `rebuild_layers.py`: 17 parts + leftover (cat ears),
  eye whites/neck restored, specks gone; live preview: 20 animated parts, all facial checks pass.
- Real run C (`004b7e89-38b2-495b-a86f-90735acaae0c`, seed 20261016, witch hat, upper body),
  after the RAM fix: **653 s end to end** (image stage ~50 s incl. load with a warm file cache;
  See-through ~10 min). App private memory during See-through 3.5 GB (was ~9 GB), free RAM at
  See-through start 7.2 GB (was ~0.8 GB). See-through's `headwear` here was a hallucinated hair
  clip (dropped, error 91); the real hat was assigned to no layer and survives as `leftover`.
- Composite-vs-source mean error inside the character after all fixes (anime-seg masks):
  A 0.91, C 0.75, dd6e1715 2.23, b6f6fe2f 1.28 (p99 16 / 14 / 86 / 25). Live preview on all four:
  19–24 animated parts, every facial diagnostic detected, no console warnings.
- `install_seethrough.py` from scratch (temp root): exit 0; the fresh engine venv imports every
  module the runner needs; the new anime-seg prefetch step works.

Ideas / next steps (owner hasn't prioritised these):

- Human checks still open: Cubism import of a package, physical webcam tracking.
- `ORDER_PRIORS` only fixes pairs seen so far; collect more real runs (full-body: legwear /
  footwear / handwear vs topwear) and extend the list.
- Persist the job queue (JSON in `outputs/`) so an overnight batch survives a restart.
- Optional: run SDXL in a subprocess like See-through (fully isolates RAM/VRAM; costs ~10 s of
  imports per job; probably unnecessary now that fact 10 is fixed).
- `run.bat` See-through prompt step and a clean-clone `run.bat` still need an end-to-end run.
- See-through sometimes assigns real accessories to no layer (cat ears, witch hat → `leftover`).
  A follow-up could split `leftover` into connected components and name them by position
  (above the head → `headwear`), so the rig and Cubism get real part names.
- The ownership margin (30) and the colour-error threshold (60) were tuned on four images.

## 9. M4 — release readiness (2026-10-17, committed)

The owner asked for "a good initial product" to push to GitHub. Done:

- **Security** (fact 20): no CORS/SocketIO, validated model ids everywhere, validated briefs,
  Host check, production + 127.0.0.1 defaults. `tests/test_api.py`.
- **Removed** the pre-M1 expression/accessory editor (+ its API, `main.js`), and modules nothing
  imported (`live2d_exporter`, `sam_segmentation`, `rigging` placeholder, `model_versioning`,
  `prompt_history`, `controlnet_expression`, `expression_animator`), `run-simple.bat`,
  `download_model.py`, `DEPLOYMENT.md`, `TROUBLESHOOTING.md` (described UI that no longer exists,
  SD 1.5, and exposing this auth-less server publicly). Dropped flask-cors, flask-socketio,
  python-socketio, opencv-python, pydantic.
- **Install**: `run.bat` rewritten (Python 3.10–3.13 via `py`, reinstall when requirements.txt
  changes — marker `.venv\kamyii-requirements.txt`, See-through offer remembered via
  `engines\see-through.skip`, `--open`); `run.sh` to match; `.gitattributes` (bat CRLF, sh LF,
  vendor byte-exact); ML stack pinned (fact 19); GitHub Actions CI.
- **Default checkpoint Animagine XL 4.0** (diffusers format, 18 files, 6.9 GB, 366 s download);
  its model-card score tags are added only for it (`MODEL_PRESETS[...]["positive"/"negative"]`).
  The Studio says "Downloading … (first use only, ~7 GB)" when the repo isn't cached.
- **Use my artwork**: `/api/generation/upload` + Studio toggle; the job imports the image
  instead of running SDXL (`GenerationRequest.reference_image`, then saved as `"upload"`).
- **Decomposition**: degenerate facial-detail filter (fact 17), leftover split (fact 18).
- **Persistent queue**: unfinished real-mode jobs survive a restart (`outputs/_queue.json`).
- **Sketches**: `sketch: true` stops a job after the artwork (status `sketch`, no preview.png, the
  Gallery/Studio show base_image). **Separate layers** = `POST /retry/<id>`, which re-queues the
  saved artwork with the original brief (seed, checkpoint, sampler…) as `source: "artwork"`.
  `GenerationRequest.source` is `prompt | upload | artwork`; the Studio hides the prompt only for
  uploads. 99 s for a sketch including a cold checkpoint load.
- **Rename**: `POST /rename/<id>` stores `display_name` in metadata.json (Gallery inline edit,
  Live Preview title); package file names never change.
- **Failures** with artwork are saved as `failed` and shown in the Gallery with the error and a
  **Retry layers** button (`POST /api/generation/retry/<id>` re-queues the artwork as an upload).
- Studio: LoRA weight slider (saved LoRA + weight restored on reopen), a notice when See-through
  isn't installed (`/model-status` → `seethrough`), reopened results select the checkpoint used.
- README rewritten with screenshots (`docs/images/`).

Verification (2026-10-17, owner's laptop):

- 37 pytest (also in a fresh torch-free venv) + 2 node tests pass.
- Real run through the API with the default checkpoint (never loaded before):
  `b2189989-a5b9-4bb1-beca-6d829aedc8c3`, seed 20261017, Euler a, CFG 5, upper body —
  completed in 981 s (a clean-clone install was competing for disk/CPU; 653 s is the quiet
  baseline). No fp16-VAE black image. Its See-through `nose` was the degenerate case of fact 17.
- All five real results rebuilt with the M4 post-processing: composite-vs-source mean error
  0.26–0.50 (p99 ≤ 15) over the composite's opaque pixels; Live Preview 19–23 animated parts,
  every facial diagnostic detected.
- Clean clone (`git clone` into a temp dir, `run.bat` with redirected input): `.venv` with
  `py -3.12`, CUDA torch + requirements, See-through installed from scratch, server healthy in
  real mode with Animagine as default. Pulling the pinning commit and re-running reinstalled
  exactly the pinned packages, skipped torch and the See-through prompt, and started.
  (Redirected input never reached `set /p` there — `fc` consumed it; fixed, see section 6.)
- Studio upload flow in the browser (re-uploading run C's artwork): toggle, thumbnail, the job
  skipped the image stage and completed in 723 s → 18 layers, the witch hat now `headwear`.
- Gallery retry on a staged failed result: queued, Studio re-attached, cancelled during
  See-through start-up; no files left in `outputs/_uploads`, GPU back to idle.

Ideas / next steps:

- Human checks: Cubism import, a physical webcam, `run.bat` prompts typed by hand, GPU runs on Linux.
- More real runs to extend `ORDER_PRIORS`; thresholds are tuned on six images. A full-body run
  (`4a66c60e-9805-4ae6-9b6d-460272d90ee9`, maid, seed 777, 806 s) decomposed well: arms L/R,
  skirt, bodice/apron, legs inpainted under the skirt; See-through put the arms under the
  clothing, now fixed by `topwear/bottomwear < handwear` priors (still image unchanged, mean
  error 0.35, p99 11; preview 25 animated parts, all facial checks).

## 10. M5 — import, edits, expressions, projects, exports, e2e (2026-10-17, committed)

The owner asked for: importing outside designs (Perchance/ChatGPT), better mouth/eye animation,
region inpainting (face, hair, new clothes; retry), extending cut-off characters to full body,
reliable import/segmentation, stable tracking, save/reopen, functional exports, e2e tests.

- **SDXL inpainting** (`ImageGenerator.inpaint`): `AutoPipelineForInpainting.from_pipe` on the
  loaded checkpoint (no second load), crop around the mask, ~1024 px work size, composite that
  blends only *outward* into the original (`max(hard, blur(hard))`; a symmetric feather left a
  ghost frame). Peak 5.3 GB on the 8 GB card. Demo mode: posterized blur fill.
- **Expressions** (`src/core/expressions.py`, job stage 0.86–0.97 and `POST /expressions/<id>`):
  closed eyes = geometric ellipse per eye from eyewhite/irides/eyelash (minus front hair), flat
  skin fill, then "closed eyes" at strength 1.0 (inpainting over the open eye left iris
  remnants); the other mouth = "open mouth, :d/:o, teeth" (or "closed mouth" when the drawn one
  is open), saved as *changed pixels only*. Prompts are face-only (character tags painted a
  beard/clothes into the patch). Preview PSD v4 maps them to the rig's `eye_close`/`mouth_open`/
  `mouth_close`; they are hidden layers in PSD/ORA. ~2 min per character.
- **Ownership fixes** (asset_separator `_resolve_ownership`): facial details beat front hair /
  headwear on near-ties (bangs were owning iris strokes); pixels no part predicts (min error >
  `ALIEN_ERROR` 90, opened with a 9 px min/max filter) are cut from every part and left to
  leftover/headwear (a witch hat's top had gone to the back hair); `inside` keeps pixels a part
  covered from being called background. Real results: holes 0.36–2 %, mean error 0.25–0.49.
- **Region edit / extend** (`regions.py`, `outpaint.py`, `POST /edit|/extend/<id>`, `GET /parts`):
  part selections use *visible* pixels only (`visible_alphas`, back to front) + a brush mask
  (PNG data URL, validated). Extend: full_body puts the art at 55 % into the top of the canvas;
  uncrop pads the touched borders 12 %; strength 1.0 then a seam pass (57 px band, 0.45). Edits
  are new sketches; empty prompt = the saved brief; each edit gets a new random seed.
- **Import** (`importer.py`, `/upload` fields `frame`, `plain_background`, `enhance` 0–0.6,
  `separate`, `crop` "x0,y0,x1,y1" fractions): anime-seg → largest connected figure → framed into
  832×1216 with a 6 % margin (kept against a border it was cut at) → white background through
  the soft alpha → optional SDXL redraw → WD tagger (`tagger.py`, wd-vit-tagger-v3 ONNX, pinned
  revision) on the *framed* image for the brief; sheet-layout tags are dropped. Overlapping views
  on a sheet join into one component — that is what the crop box is for.
- **Projects**: Gallery Export zip → **Open exported character** (`POST /open-project`,
  `project.py`): zip-slip/size checks (bytes counted while writing), staging under `_uploads`,
  a new id when the old one exists, absolute paths in every JSON rebased to the new folder.
  Global `MAX_CONTENT_LENGTH` is 1 GB but a `before_request` hook keeps everything except
  open-project at 16 MB (Flask 3.0 has no per-request limit).
- **Studio adjustments** are saved with the character (`PUT /studio/<id>` → `studio.json`,
  injected as `KamyiiPreview.settings`, applied with `anyModel` because the PSD fingerprint
  changes whenever the preview PSD is rebuilt). Plain PSDs opened in the Studio still use
  localStorage.
- **PNGtuber pack** (`pngtuber.py`, `artifact/<id>/pngtuber`): character.png + expression layers
  → idle/talk/blink/talk_blink PNGs cropped alike, with a README for veadotube mini, PNGTuber
  Plus and Discord overlays. Shown when the character has expressions.
- **pytoshop** wrote a NUL into every PSD layer name; patched `encode_unicode_string`.
- **e2e** (`tests/e2e`, Python Playwright): three browser journeys against an in-process demo
  server; skipped without a browser. Locally: `KAMYII_E2E_CHANNEL=msedge` (playwright 1.47 is in
  `.venv`, no bundled browser). Edge's download UI crashes the 1.47 driver on close, so the
  tests fetch the Export link with `page.request` instead of a real download. CI runs them on
  Ubuntu with bundled Chromium.

Verification (owner's laptop, RTX 4070 Laptop 8 GB):

- 65 pytest + 3 e2e (Edge) + 2 node tests.
- Inpaint/expressions on real results: eyes close cleanly, mouth opens (116–126 s per job).
  Hair recolour edit touched only the hair; full-body extension of a hoodie character had no
  visible seams (~1.5 min).
- Import of outside-style art: a 512 px neon-street JPEG and a reference sheet came out as framed
  white-background sketches with sensible tag briefs. The sheet's front view, boxed with the crop
  (`ab1ffc38…` sketch → `4bb6ee7b-e6ea-4885-a91b-f8ea3b305eb5`, 881 s): 20 layers, composite
  mean error 1.18, 0 % holes, expressions painted, PNGtuber states look natural.

- **Tracking dropouts**: a camera frame without a face used to set `cam.live=false` at once, so
  the avatar jumped toward the idle pose and back at every brief loss (a hand, motion blur).
  Now a missing face holds the last pose; the existing 1.2 s window hands back to idle.
  Synthetic still face that vanishes 0.2 s every 1.5 s (headless Edge, steady state): before,
  15 % of frames fell back to idle, angleX std 0.188, per-frame jumps 0.27; after, 0 %, 0.011,
  0.031. The re-detection after a loss takes FaceMesh a few hundred ms, hence the hold.
- **Outline matte** (`AssetSeparator._clip_to_matte`, anime-seg only, also on character.png):
  See-through's parts reached a few px into the white background (1,258 opaque white px on the
  imported maid) — a glowing outline over a dark stream backdrop. Within 4 px of the outside,
  parts are clipped to the matte, only the topmost solid part keeps the soft edge (stacked parts
  at the matte's alpha composed darker than the source) and the plain background colour is
  un-mixed from it. **The matte reads 250-254 across most interiors**: clipping there made every
  layer ~98 % opaque and the owner rule would have erased hidden parts, so the interior is never
  touched. All eight real results were rebuilt: nothing opaque outside the matte, interiors
  identical to a no-clip build, composite-vs-source error equal or better except 889 sleeve-edge
  pixels on `4a66c60e` (which look cleaner on dark). (Interiors at 250-254 alpha are See-through's
  own; harmless.)

Not done / next:

- Tracking stability: dropout hold added (above); smoothing itself was already a One-Euro filter
  with calibration; a physical camera is still untried.
- Physical webcam/mic, Cubism import and Linux GPU runs still need a human.
- Expressions only for See-through characters; demo mode has none (the demo rig blinks by
  squashing). Mouth shapes are two (closed/open); no vowel shapes.

Owner-reported issues (2026-10-18), addressed in the preview motion polish below:

- **Blink**: the painted closed eye sometimes "rises" and the two eyes look uneven. The painted
  layer is `expressions/eye_close.png` (one ellipse per eye, `expressions.eye_mask`), inserted as
  the rig's `eye_close` after eyewhite/irides/eyelash (`live_preview._with_expressions`); the
  vendored rig fades it in per side. Fixed in `static/vendor/anime25d/lib/rigger.js` and `app.js`; details below.
- **Hair physics**: shadows the front hair casts on the face don't follow the swaying hair.
  Confirmed baked into See-through's face/skin layer; preview shading now shares the bang displacement.

Preview motion polish (2026-10-18, uncommitted):

- Read the **Ka-myii Progress Update** chat: importing characters, painted edits and better
  facial movement are the core workflow; the first two already shipped in M5. This pass
  focuses on motion rather than adding a new feature set.
- Closed eyes previously used the alpha centroid of a broad skin patch as their anchor,
  scaled around the open-eye centre, rose by `eyeOpen * 3` during the crossfade, and had a
  different parallax depth from the skin. Closed-position anchor edits only moved the open
  eye's squash point. Anchors now detect dark lid ink per side; closed art scales/rotates
  around that lid and follows manual close-position edits. Facial ink/skin share depth 1.
- Natural auto-blink: 80 ms close, 35 ms hold, 120 ms reopen, with faster eyelid response so
  it still reaches full closure. Optional long-closed variant has a 650 ms hold (was 3.4 s).
- Front-hair spring displacement saturates smoothly at 18 * faceScale pixels, keeping roots
  fixed and limiting violent motion on rapid turns. Back hair/accessory tuning is preserved.
- The renderer transports skin textures' baked shading (including painted closed-eye patches)
  using the same strand field and bang controls as front hair, including its extra head parallax. A denser face mesh supports
  that field. Eyes/brows/nose/mouth and a rim around the face are protected; alpha stays fixed.
  **Hair shading → Shadow follow** (0–1, default 1) lets the owner reduce the effect for unusual
  painted lighting. It participates in save/restore, reset and undo. Saved old automatic
  facial depths migrate; deliberate custom depth edits are retained (`rigRevision: 3`).
- Preview PSD **v5** fills ghost hair/eye/brow ink under fully opaque overlying parts with
  locally interpolated skin. This removes duplicate marks revealed by motion. The original
  rest composite and alpha stay identical; artist PNGs, PSD/ORA and Cubism exports are untouched.
  No GPU regeneration or new dependency is needed. The new cache is built lazily per character.
- This is local transport of existing painted shading, not a 3D lighting solver. Artwork
  with shadows far outside the fringe can still need artist refinement. Physical camera/mic,
  human Cubism import and clean-install checks remain outstanding as above.

Verification:

- **70 GPU-free pytest tests pass**, including source immutability, unchanged rest composite,
  unchanged alpha and safe handling of translucent overlying artwork.
- Both Node suites pass. Added lid-vs-skin centroid, close-anchor movement, scale/rotation,
  attached roots, bounded bangs and spring stability at 10/30/60/240 fps checks.
- Focused Edge browser regressions pass: shading/anchor adjustments survive reopening;
  old default depths migrate while a custom mouth depth stays unchanged; the real shader
  shifts a painted stripe by 8 px while preserving every alpha pixel and reporting no GL error.
  The Gallery export/reopen journey passes too. An earlier broad browser run passed its first
  two journeys but had driver-startup errors on the next two; fresh targeted processes passed.
- Real-character headless Edge check: witch `004b7e89`, silver hair `b6f6fe2f` (generic closed-eye
  fallback), imported maid `4bb6ee7b`; combined turn/nod/tilt, both eyes closed, bang movement,
  Shadow follow on/off, and automatic blink cycles. Each reached eye openness 0.03 and 60 fps;
  no console warnings/errors. The final v5 skin-cleanup previews were also checked on all three,
  including Shadow follow on/off, with 60 fps and no errors. Camera/mic remained off. Screenshots/reports: ignored
  `outputs/physics-review/`.
- Real web server on localhost:5055 was reloaded while its job queue was idle, without a
  reloader. The separate dataset generator was left running; no generation job was started.

Manual blink editor (2026-10-18, uncommitted):

- The owner’s `82c64229-de47-435c-bce7-759724a5e096` still has uneven painted lids and copied
  details in its eye patches. Automatic rig anchoring cannot fix a badly placed drawing.
- **Position blink** on the stage / **Blink alignment** in the panel freezes a neutral head,
  holds both eyes closed, and zooms to the face. Drag either eye box or enter independent
  horizontal/vertical offsets; arrow keys nudge 1 px (Shift 10). Left/right refer to image sides.
  Each eye has its own size and tilt. Open/both-closed/wink views support comparison.
  **Match open eye line** places each lid at 62% of its own open-eye height as a starting point.
- Close anchors now accept `dx` and `dy` (old vertical-only settings still work). The deformation
  helper translates around an independent target X, so horizontal edits really move the art.
  Global close-height/angle controls remain compatible with older adjustments.
- **Blink artwork → Clean fallback blink** reversibly rebuilds the browser rig using the existing
  tinted generic lids instead of the painted eye patches. This avoids copied brow/hair details
  without a GPU rerun. Original painting remains available. Changing artwork resets only blink
  alignment; Undo restores the previous artwork and placement. Generic mouth behaviour is unchanged.
- **Test motion** exits alignment and plays; **Reset blink alignment** resets only placement,
  size and tilt; **Save adjustments** keeps all settings in the existing `studio.json` and project
  export. Temporary held-closed/editor states are not saved. Artwork choice also survives reopening.
  Artist PSD/ORA/PNGtuber exports stay as before; preview PNG/video use the adjusted rig.
- Preview PSD **v8** additionally heals full eye sockets and a small halo around translucent
  facial parts. This model had transparent stroke-shaped holes in its face: dark back hair
  showed through as old eye/brow outlines after the open parts faded. Repaired sockets receive
  opaque interpolated skin inside the face silhouette; original artist alpha is untouched.
  This supersedes v5's preservation of every preview-alpha pixel: interior eye holes now close,
  while the outside silhouette stays intact. Three float-precision separable box passes avoid
  colour rings caused by dividing quantized 8-bit blur masks. No new dependency.
- No SDXL generation/pipeline was changed or started. The existing `/expressions/<id>` endpoint
  still repaints with the saved character seed; it is not a random retry UI.
- Verified: Node regression checks include XY translation, old/new anchor validation and clean
  fallback replacement. **73 GPU-free pytest tests and both Node suites pass**; all three focused
  Edge regressions pass. These check actual rendered lid vertices moving 7/-9 px,
  independent eyes, keyboard nudges, Undo/Redo, saved positions/tilt/artwork, reset, matching eye line,
  temporary pose exclusion and a 390px layout. Also rechecked earlier shading/settings behaviour.
  On the owner’s exact model, pointer dragging and Undo worked; original/matched/clean views and
  phone layout were inspected in Edge without console warnings/errors. Screenshots in ignored
  `outputs/blink-review/`. The clean fallback removes copied patch detail and repaired eye-hole
  outlines; changing placement alone cannot repair the pixels in a damaged painted patch.

Roadmap (owner): once emotions are right, emotes such as meme faces and waving.

LoRA dataset (2026-10-18): `scripts/build_lora_dataset.py` generates varied designs with the
default checkpoint and keeps rig-ready ones (one figure, head not cut off, plain white background,
no stray ink, WD-tagger checks for eyes/hands/pose/text/rating), captions `kmyvt, <tags>`, writes
`datasets/<name>/images` + a zip + contact sheet + report.json (rejection reasons). Resumable: rerun
the same command. Trial: ~40 s per candidate; "upper body" alone often cuts the head at the top,
hence 50 % cowboy shots. A 320-candidate run (`datasets/vtmodel`) was started 2026-10-18.
The matching SDXL trainer notebook is outside the repo (owner's
`Downloads/image_checkpoint_trainer_sdxl.ipynb`): bucketed, cached latents/embeddings, Ka-myii's
chunked prompt encoding. The owner's first notebook run (2026-10-18, 56 images from the 320-candidate
set, 896² buckets) trained on the 8 GB laptop at ~5.5 min/epoch, saving epoch files as expected;
notebook progress bars don't render there (tqdm.auto without ipywidgets in `.venv`).
`scripts/train_lora.py` is the same method as a console script (per-epoch bar + summary, CSV log,
`--resume`, `--samples`, installs into `models/loras`); GPU-free tests in `tests/test_train_lora.py`,
not yet run on the GPU itself. The `.venv` has `ipykernel` and a "Ka-myii (.venv, CUDA)" Jupyter kernel.

Preview speech, bounce and companions (2026-10-23, uncommitted):

- **Fluid mouths**: `motion.js` supplies continuous compressed-open shapes and a wide, gradual
  artwork blend, replacing the old abrupt threshold/half-open minimum. Lip response has separate
  attack/release rates independent of frame rate; Rounded shape narrows open art. Demo speech
  varies rounding and can talk while preserving a selected expression (e.g. Smile). Camera/mic
  still supply openness; no phoneme recognition is claimed.
- **Bouncier motion**: Gentle / Lively / Bouncy buttons and a saved Bounce slider. Stronger idle
  head/body movement, shared model translation for follow-through, livelier strand wind and springs.
  All facial parts share the same final translation. Front hair remains smoothly bounded at
  `(18 + 4 * Bounce) * faceScale` px; other strand shifts at `60 * faceScale`. Hair shading uses
  exactly that same field. Tracking retains 85% of hair amplitude, instead of halving amplitude
  and softness together. Idle off now stops breathing as well; Physics off stops follow-through,
  ear/tail motion and bust jiggle.
- **Nekomimi & tail**: optional, initially off, available on existing characters without ML or
  regeneration. Original canvas-drawn ear/tail textures live in `lib/motion.js` (MIT); fur matches
  sampled hair or Ivory / Charcoal / Copper. Rooted ear meshes twitch independently, a continuous
  tail ribbon curls/swishes with inertia, and both follow their head/body attachment transforms.
  The tail starts behind the torso rather than below a cropped portrait. Controls: ear size,
  spacing, height, twitch; tail size, XY, side, swish. These are numeric settings using the
  existing validation, save/reopen and Undo/Redo paths. No additional device permission or service.
- Deliberately named `cat ears` / `nekomimi` PSD parts (including left/right suffixes) get separate
  ear motion. Human `ears` stay rigid; `cat tail` aliases `tail`. Ears baked into an inseparable
  headwear/hair layer are **not automatically extracted**. Turn fitted additions off on characters
  already painted with them to avoid duplicates. Existing artist tails retain their strand sway.
- Preview-only additions are included in preview PNGs and recordings, not artist PSD/ORA,
  PNGtuber or Cubism exports. Transparent stage padding and a renderer `uOffset` retain PSD/anchor
  coordinates and keep fitted ears/tails visible, including motion margins. Fit, Face detail and
  anchor dragging account for padding; recording resolution stays fixed during capture. Padding
  respects WebGL texture/viewport limits and a 24 MP budget (extreme fits can clip at those limits).
- README and vendored UPSTREAM notes updated. No weights, original model layers or owner studio
  settings were changed. A real-mode server was started at 127.0.0.1:5055 (reloader disabled,
  PID 24544; job queue checked empty before restart). Review screenshot:
  `outputs/motion-review/final-preview.png` (ignored).

Verification:

- 76 GPU-free pytest tests passed; both Node suites passed, with added tests for continuous mouth
  alpha/shape, lip attack/release at 10/30/60/240 fps, rooted/bounded ear/tail geometry, and named
  animal-ear PSD splitting (human ears unchanged).
- Four focused Edge journeys passed: existing blink editor, shading/settings, real WebGL shadow
  shader and the new speech/companion journey. The latter checks actual renderer vertices/alpha,
  independent motion, stopping motion, rapid head turns (finite meshes), colours and fit saved
  across reload, Undo, 390px overflow, and an actual transparent PNG with padded dimensions.
- In-app browser: inspected the owner's silver/blue character with fitted ears and attached tail,
  Bouncy motion and demo speech. A screenshot showed 55 fps (single observation, not a benchmark).
  Camera/microphone stayed off. Physical-device feel and a fresh video recording remain untested.

Reference-style springy nod (2026-10-23, uncommitted):

- Owner supplied `Downloads/nina-nina-saotome.gif` as a movement reference (27 frames, 0.91 s).
  Inspected sampled frames: rapid head-led dips/rebounds, quieter shoulders and trailing hair.
  It is a reference only; its character artwork is not bundled in the app.
- **Body & physics → Springy nod** enables depth .85 / speed 2.6 cycles/s, Idle and Physics.
  Nod depth (0–1.5) and Nod speed (0.4–3.4 cycles/s) use existing settings/save/Undo; old settings
  default to depth 0. Gentle/Lively/Bouncy clear the repeating nod. A continuous phase accumulator
  avoids phase jumps when adjusting speed; depth eases in/out. Camera mode uses the user's nod
  and never adds the repeating idle cycle.
- Head-group parts share the same Y translation and small compression around the neck pivot.
  The neck blends from full head movement at the jaw to the quieter torso. This keeps eyes,
  lids and mouth on their skin plane, including old blink adjustments. Body movement is much
  smaller and phase-lagged. `motion.js` holds the pure nod curve/point helpers.
- Every strand now has vertical stiff/soft springs too. Roots stay fixed; tip travel is capped
  at 8 * faceScale px for front hair and 26 * faceScale for other parts. Hair skin-shadow UVs
  use the exact same vertical offset. Canvas margins include the nod and optional ears.
- Both Node suites passed, with new nod/torso ratio, compression, rooted vertical bang limits
  and spring checks at 10/30/60/240 fps. Three focused GPU-free Edge journeys passed: actual
  rendered head-vs-torso excursion, nod settings survive reload, regular presets clear nod,
  existing companion/export flow, blink editor and hair-shadow/settings checks. Python backend
  is unchanged (76-test baseline passed earlier this session).
- In-app review on the owner's silver-haired model: no console warnings/errors; real preview
  recording exported at 832×1292, 3.1 s, ~1.4 MB. Copy and screenshot in ignored
  `outputs/reference-motion/springy-nod.webm` / `springy-nod.png`; sampled reference frames there
  too. Owner's saved studio settings were not overwritten. Physical camera nod feel remains
  untested. Server restarted with empty job queue, reloader off, port 5055, PID 18728.

Neck movement and head tilt (2026-10-23, uncommitted):

- Owner: "head movement is good but there is zero neck movement"; references
  `Downloads/tenor.gif` (side-to-side head sway, ~1 cycle/s) and `Downloads/no-vtuber.gif`
  (head shake over a still choker). Cause: roll was `angleZ*0.07` rad (4° at full) and the
  tracker's roll gain 1.4, so a real 20° tilt rendered ~2°.
- Roll is now `Motion.ROLL` = 0.28 rad (16°) at |angleZ| = 1; tracker gain 3.6 ≈ 1:1. Idle tilt
  a little larger. `Motion.rollPoint`: rigid roll about the Neck pivot above the chin; below the
  chin a point slides like the chin line and its lift fades over one face height (cut-off hair
  never rises off the canvas bottom). `neck`/`neckwear` blend from full at the chin to none at
  the Neck pivot (the existing anchor handle now also sets where the bend ends). The torso no
  longer rotates about the neck (16 % share removed); it leans 0.4 × body range about the hips.
  Hair springs get the physically correct lateral roll displacement.
- **Head sway** button/slider (`swayDepth`, 0–1): tilt loop at half the nod tempo (nod
  accumulator now wraps at 2), soft-square so it lingers at each side; presets clear it.
- Canvas padding for a full tilt uses opaque row extents (`Motion.rowExtents`), skips head
  layers already cut at the top/sides (their cut would show anyway) and bottom-edge points.
  Owner's model: no padding; witch `004b7e89`: 42 px. Cut-off art (that witch's hat) shows its
  straight cut edge at full tilt; Extend (outpaint) is the fix.
- Preview PSD **v10**: the neck under opaque neckwear gets interpolated skin, and clothing
  under neckwear, above the default neck pivot and over an opaque neck, is made transparent.
  See-through had painted a grey copy of the choker and its star into the hoodie on
  `82c64229`; a moving choker revealed it. A faint outline of the old pendant remains at its
  translucent edges. Artist PSD/ORA/Cubism untouched.
- Verified: 77 pytest, both Node suites (roll/neck/sway/outline/tracker-gain checks), 4 Edge
  journeys incl. a new rendered check (face rolls 0.26–0.32 rad, torso < 0.03 rad, sway reaches
  ±0.5, presets clear it). In-app on `82c64229` and `004b7e89`: ±16° tilt, neck bend, choker
  follows, sway ≈ 1 Hz, no console warnings. Physical webcam tilt not tried. Mouth not changed.
  Server restarted with `.venv` (empty queue), port 5055.

Shoulders, collar, hats and full-body extension (2026-10-23, uncommitted):

- Owner: "add some more movement to the body like moveable shoulders", "the shirt collar to
  shift with the neck", the witch's hat "is static", and "extend these models to give them arms
  and legs if only the top was generated", with VTuber model sheets as stance references
  (`Downloads/images.jpg`, `*reference-sheet*.webp/png`: standing straight-on, arms relaxed).
- **Collar**: `Motion.collarWeight` replaces `neckWeight` for every body layer: 1 at the chin,
  0.35 at the Neck pivot, fading over 0.3 face heights below it, and sideways from 0.22 to 0.44
  face widths off the neck's centre line. Roll, head XY and nod use it, so the collar, choker and
  neck base move together; shoulders beside the neck stay. Body layers now use a 24 px mesh.
- **Shoulders**: `Shoulder shrug` (pose, −1…1) and `Body motion` (0–2, default 1) sliders.
  `shoulderLift`: no lift within 0.25 fw of the centre, full past 0.8 fw, fading below the
  Neck pivot over min(1.3 fh, 85 % of the space to the canvas bottom) so cropped chests never
  leave the bottom edge; `handwear` (arms) rises whole. 20 × faceScale px at 1. Body motion adds
  breathing (0.2), the tilt-side shoulder (0.5 × |tilt|^1.5) and an Idle roll out of step (±0.16).
  `turnShift`: head turn moves the torso's centre line 6 × faceScale px, its outline 30 % of
  that, nothing below the hips (Neck pivot + 1.4 fh) and nothing where the collar follows the head.
- **Hats**: `headwear` trails the crown (face top) on its own X/Y springs (k 42/55, gain 2.6/2,
  limit 0.12 fh, × Accessory sway, Physics only, small Idle wobble). `hatWeight` is 0 within half
  a face height of the crown (hair clips, the band) and 1 beyond one. Witch, full turn: brim
  corner lags ~8 px, overshoots ~9 px, settles in ~1 s. Her hat is split across headwear, front
  hair and back hair by See-through; the seams along the brim come from layer parallax that
  existed before, not the lag. Stage padding allows for the lag on uncut hats.
- Preview PSD **v12**: topwear/handwear under opaque front/side hair gets interpolated cloth, and
  thin transparent cracks next to the hair (the separator gave the hair's outline pixels to the
  hair) are closed with a 5 px morphological close. On `82c64229` a shrug had revealed dark lines
  along the strands; they are gone. Faint pale strand copies outside the hair remain.
- **Extend to full body**: `outpaint.plan(..., head=)` scales the art so the figure is ~5.5 heads
  (`HEADS`), head = 1.1 × the separated face layer's height (`head_height`); sketches without
  layers keep 0.55. "upper body"/"waist up"/"portrait"/"cowboy shot"/… are removed from the brief
  (`full_body_prompt`, also the new character's saved brief); tags now ask for a straight-on
  standing pose with arms at sides and forbid from above/below, sitting, multiple views. Edges cut
  at the top/sides are repainted 12 % in (bottom join stays 5 %). Live Preview → Body & physics
  links to `/generator?model=<id>#extend`, which opens the panel at the button.
- Real GPU runs on the witch `004b7e89`: v1 (5 % side band) → `3cd80ebb` — natural proportions,
  full hat, shoes, but grey blocks of the old brim/sleeve edge at the art box's sides; the 12 %
  band was added for that; v2 → `a03c0f75` — clean, straight-on, arms relaxed, shoes, hat redrawn
  smaller. Both are unseparated sketches in the gallery (owner may delete them).
- Verified: 78 pytest, both Node suites (collar/shoulder/turn/hat helpers), all 7 Edge journeys
  (new: tilt moves the rendered collar >4 px beyond the torso lean; Shrug lifts the outer shoulder
  >5 px, not the collar or the cropped bottom). In-app: `82c64229` collar at full tilt, shrug
  before/after v12; witch hat lag measured; full-body `4a66c60e` shrug lifts arms ~6 px, head
  turn moves feet 0.5 px. Studio `#extend` link opens the panel. Physical webcam untried.

Torso follow, expressive mouths and discoverable full-body completion (2026-10-23, uncommitted):

- Owner requested research into natural VTuber torso movement, more body movement, teethy smiles,
  shocked O/jaw-drop mouths, and an easy way to complete existing partial models. Official Live2D
  motion/body/vowel/parameter references and our design choices are linked in `docs/MOTION_NOTES.md`.
- `Motion.torsoPoint` is a shared parent arc, replacing the tiny rigid whole-model lean. Separate
  yaw/roll/lift springs follow head motion; smooth shoulder-to-hip falloff roots the lower body
  and cropped canvas bottom. Head, collar, clothes, arms and companions inherit the parent;
  hair springs include its displacement. Idle roll is stronger and slower than head detail.
  **Body turn** poses yaw directly; **Body motion** sets the inferred/idle response. Camera still
  tracks the face, not a real full-body skeleton. Stage margins include torso extremes on uncut
  head art and companions without padding already-cropped hair into artificial straight edges.
- Original canvas artwork (no downloaded assets) adds **Toothy smile**, **Surprised** (O) and
  **Jaw dropped**. Mouth artwork / Show teeth / Rounded shape / Jaw drop controls, normalized
  crossfades, mouth-ink-derived placement, editable mouth anchor and existing scale settings.
  Switching presets clears old shape/jaw flags. **Original only** uses the painted source mouths.
  Settings/undo and PNG/recording exports include these shapes; artist PSD/ORA/Cubism do not.
- Preview PSD **v13** additionally heals facial mouth sockets/holes beneath replacement shapes.
  It rebuilds lazily from the original aligned assets; original model artwork is untouched.
- **Full body ↗** is now in the preview header. Studio has **Complete body** beside the result
  and an independent **Complete the body** panel (legacy `#extend` still opens it; new `#fullbody`).
  Optional outfit details supplement the identity brief. Result choice: **Full model + Live Preview**
  (draw → layers → expressions → exports) or **Artwork to review first** (sketch → Separate layers).
  `/extend` accepts validated `separate: true`; omitted/false keeps its old sketch behavior.
  Demo mode explicitly explains its placeholder artwork. The result URL follows the new model
  so refreshing/bookmarking no longer reopens the parent. All extensions create new characters.
- Verified: **80 GPU-free pytest**, both Node suites; four focused Edge journeys passed for
  hair shading, manual blink editor, fluid speech/companions/torso, and both full-body result
  choices. The final torso-margin browser check passed separately. New checks cover normalized
  mouth blends, mouth ink versus broad patch placement, old mouth socket healing, rendered
  teeth pixels versus O, saved jaw-drop presets, rooted torso/feet, identity preservation and
  full-model extension outputs. Reload-after-extension is checked in the full-body journey.
- Real GPU full-body artwork run from owner's `82c64229-de47-435c-bce7-759724a5e096`:
  **`964575da-f907-4d94-8623-3e625c97a599`**, seed **568137086**, novaAnimeXL, 832×1216,
  28 DPM++ 2M Karras steps, **141.0 s** including cold model startup and seam blending.
  New copy has the white/blue hoodie, a pleated skirt, tights and sneakers; front-on arms,
  legs and feet visible. It is deliberately an **unseparated sketch** for owner review, with
  Separate layers available. The automatic full-model route was verified in demo-mode tests,
  not repeated on GPU. Highest sampled GPU memory **6965 MiB** (not a guaranteed peak).
- In-app review: teeth and O fit the silver-haired model without a second mouth; no console
  warnings/errors in preview or Studio. Screenshots are in ignored `outputs/torso-mouth-review/`
  (`toothy-smile.png`, `jaw-dropped.png`, `full-body-studio.png`). Owner's saved preview settings
  restored after inspection. Physical webcam feel and Cubism human import remain untested.
  Server restarted only after confirming no running/queued jobs; real mode, 127.0.0.1:5055,
  reloader off; launcher PID **20976**, Python child **17504**. Logs in the review folder.

Seamless full body and expression toggles (2026-10-23, uncommitted):

- Owner (screenshot of the three full-body sketches): "the full bodies are pretty misaligned …
  try making the top to full body look seamless" and "begin working on adding a generation option
  for toggles like blushing and other expressions and like emotes".
- Diagnosis: the joins themselves matched, but the hard art box plus fixed bands made SDXL
  continue cut structures: the witch v2's cut brim edge became a hanging ribbon, v1 kept grey
  blocks, the hoodie (`964575da`) got a translucent ghost strand beside the box, and every result
  had a faint grey frame line at the right edge.
- **Graded extend mask** (`outpaint.plan` now returns uint8 0…255): 255 on the new canvas, 0 deep
  inside the art, linear ramps along its edges (cut top/sides: full for 6 % of min(w,h), to 0 at
  25 %; bottom join: 3 % → 20 % of h; uncut edges 4 %). `ImageGenerator.inpaint(graded=True)` is
  differential diffusion through the SDXL inpaint pipeline's `callback_on_step_end` "mask" input:
  `free_during(change, step, n)` lets a pixel with change c be redrawn for the last c share of the
  steps; the final composite alpha is min(1, 4c) (no half-transparent double lines). The 57 px
  seam pass is gone (one SDXL pass, ~25 s saved). "border, frame" joined the full-body negative.
  Edits keep hard masks.
- Real GPU, same seed as `964575da`: `4a46a1a0` — no ghost strand, the hoodie flows into the body;
  witch → `e90d32a8` — one coherent wide-brim hat, no ribbon/blocks; right edge flat (no frame).
  Both are unseparated sketches; the witch's background came out light grey (240).
- **Expression toggles** (`expressions.TOGGLES`): Blush ("heavy blush", 0.7: many characters
  already blush lightly) and Tears (cheek regions under each eye, on skin only), Heart / Star eyes
  (eye ellipses, 0.7), > < and Spiral eyes (eyes flooded with skin first, 1.0, like closed eyes).
  Sweat drop (image-right temple) and Anger vein (image-left forehead, over the hair) are drawn
  vector symbols (`draw_symbol`, 4x supersampled): painted by SDXL on the witch, "anger vein"
  became a tiny girl and "sweatdrop" changed nothing. `paint_toggles` writes
  `expressions/toggle_<id>.png` and `expressions.json["toggles"]`; `generate()` keeps them.
  Studio brief chips (`toggles` in `/generate`, validated), `POST /expressions/<id>` with
  `{"toggles": [...]}` paints only those (no eye/mouth repaint), Live Preview **Paint toggles**
  (See-through characters) queues that and reloads when done. PSD/ORA: hidden `toggle <id>`
  layers. Preview PSD **v14**: `toggle <Label>` layers under the front hair (over-hair ones on top).
- Rig: `toggle …` names get `TOGGLE_SLOT` (head, depth 1, `fade:'toggle'`); hidden by default
  (also for old settings and Reset layers); visibility is the switch, so undo/save/OBS work.
  Buttons under Expressions; Shift+1–9.
- Verified: 83 GPU-free pytest, both Node suites (toggle slot), all 8 Edge journeys. GPU: all
  eight toggles on the witch `004b7e89` (~12 s per painted toggle, ~4 min job incl. cold load);
  hearts/stars/> </spiral/tears clearly right, blush subtle on her already-blushing face (prompt
  then strengthened and repainted). In Live Preview the buttons start off, combine, follow head
  turn and tilt, Shift+7 works, no console warnings. Studio chips send `toggles`. Server
  restarted on 5055 with no jobs queued (PID 25192).
- Not done: emotes that change the pose (waving), per-eye toggles that follow blinking (an eye
  toggle covers the eyes while on), uploads' import form has no toggle chips, PNGtuber pack
  ignores toggles.

Guided extension quality, natural sway and blink-aware emotes (2026-10-23, uncommitted):

- Owner asked to keep improving output quality, especially large heads, body alignment and
  seamless extensions, using real VTuber movement references, and to leave further ideas.
  Read all prior handoffs and preserved the uncommitted earlier work/trainer files.
- **Proportions and fit:** `outpaint.anatomy` measures crown-to-chin hair occupancy (ignores
  sparse ahoge tips), uses neck centre for the standing axis, and retains measured geometry
  on artwork-first results. Balanced **6.75**, Tall **7.5**, Compact **5.5** are artistic targets.
  Large headwear gets 12 % headroom instead of 4 %; ankle targets leave 0.45 head heights for
  shoes below them. The read-only `/extend/<id>/plan` and Studio **Preview the fit** show source
  placement, ruler and shoulder/hip/ankle targets. Original character files are preserved.
- **Masking alone was insufficient:** four unguided small-head trials left a pasted chest or
  a second head. They are labelled "Alignment experiment … - superseded" in the Gallery:
  `8f290a89`, `eab60823`, `24371d1a`, `2d168bcc`. They are retained for comparison, not quality
  examples. The first graded diffusion mask now uses the actual first-step change threshold
  rather than all nonzero grey pixels; subsequent masks advance by one step and update in
  place (the 0.37 ControlNet pipeline exposes `mask` but ignores replacing it in its callback).
  Retained head alpha is hardened (See-through interiors are often ~250, not 255), preserving
  one coherent head, while the lower outfit is redrawn continuously below the collar.
- **Standing-pose guidance** is the recommended/default full-body path (`guided: true`;
  optional off for experiments/SD1.5). Original OpenPose-format lines are tied to the measured
  head, neck, shoulders, hips and feet; no OpenCV/pose-detector dependency. Uses
  `xinsir/controlnet-openpose-sdxl-1.0`, Apache-2.0, pinned **23f966cd5cfdd3f7729c903e243d87152162d2b7**,
  ~2.5 GB downloaded to HF cache on first use. README credits it. No ComfyUI dependency,
  package upgrades, source checkpoint changes, weight commits or model assets vendored.
- **Critical 16 GB RAM recipe:** encode the prompt FIRST; remove all CPU-offload hooks, then
  register both SDXL text encoders as None and collect/free them BEFORE loading ControlNet.
  Store UNet/ControlNet weights in FP8 with FP16 compute using existing diffusers layerwise
  casting; normal offload (no NF4/group offload). Enable tiled VAE decoding. The pose operation
  caps Torch's caching allocator at min(previous fraction, **0.86**), clears unused blocks per
  step and restores the prior fraction on exit. Delete the temporary pose pipeline and unload
  the generator afterwards, so later jobs reload the checkpoint's original precision.
  Keeping encoders through loading/decode caused severe paging on two trials; those were
  cancelled/stopped (only our server) and their pending queue entries backed up then removed
  (`outputs/fidelity-review/cancelled-pose-trial*.json`). No owner job was cancelled.
- Real guided hoodie: **`a834d4e3-2c0e-4f00-b3f9-d09cbc7606c3`**, seed 568137086, novaAnimeXL,
  832×1216, 28 requested steps / 25 denoising (strength .9), **185.3 s** with the earlier cache
  recipe. Clean collar/body connection and retained face. It remains an artwork sketch.
- Real guided witch, final framing/cap: **`42991481-8bb7-450d-8f74-f02c023dd0ae`**, seed 809031280,
  same checkpoint/size/steps, **97.3 s** including cold startup. Whole hat and boots fit;
  aligned dress/arms/legs. Measured Torch peak **4301 MiB allocated / 6744 MiB reserved**;
  GPU returned to **175 MiB** idle after unload. Earlier uncapped guidance reserved 7976 MiB,
  so do not remove the allocator bound. Prior witch `03bdfecd` is a superseded framing test.
  These are sketches for review. Guided generation was verified with novaAnimeXL, not every
  checkpoint/LoRA combination; unguided large extensions can still drift.
- **Complete animated deliverable:** separated the clean hoodie sketch without re-generating
  its artwork → **`c2e836b7-002b-4a59-8ef7-7c5dddae4e67`**, display name
  **Study - balanced full body (animated)**. Real See-through + painted blink/talk + exports
  completed in **1223 s** (job wall 1224 s). Added the two new drawn emotes afterwards: **28
  exported assets** including hidden expression layers. In-app full-body fit and Face detail
  wink checked; the blink stays on the eye line. Saved **Natural sway** on this new copy
  (weightShift .9, bodyMotion .85, breathing 1). All six facial detections pass; the preview
  diagnostic reports generated tinted closed eyes despite an exported painted eye-close file
  (simpleBlink=0). The rendered wink is clean; investigate fallback selection before claiming
  that painting is used in this preview. Original model's saved settings were restored.
- **Natural sway:** quintic transfers with a quiet dwell, hip-to-foot falloff (fixed soles),
  slower inhale/exhale and adjustable Weight shift / Breathing depth. All head/collar/clothing
  inherit the same parent. Camera mode does not add the idle weight-transfer loop. Studied
  Live2D's actual Jin Natori video (body from 11:01, secondary motion 11:47) and body tutorial;
  the prior Nina GIF still informs opt-in Springy nod. References and limits in
  `docs/MOTION_NOTES.md`; no third-party reference art/video is shipped.
- **Eye emotes:** heart/star/spiral overlays split into independent eyelid-aware halves,
  grouped as one switch; > < suppresses ordinary open/closed eye art. Competing eye designs
  are mutually exclusive; blush/symbols still combine. Fractional layer order and migration
  preserve older single-layer toggle settings and following artist-layer IDs.
- **Gloom / Sparkle aura:** original supersampled manga overlays; no diffusion/GPU needed.
  Gloom clips to the face, sparkles float/pulse with Idle. Existing-model vector-only jobs
  skip checkpoint loading. Added both to the original silver-haired model (5 s) and the new
  full-body copy. Hidden layers in artist PSD/ORA; saved preview controls/PNG/video work.
- **QOL/security:** do not inherit staged `pose_image`/mask/source paths from saved briefs;
  retries clear the pose path. Emote repackaging re-reads current metadata before patching
  exports so a rename made during the job is preserved (observed/fixed on the new copy).
- Verification: **90 GPU-free pytest**, both Node suites; **all 8 Edge journeys passed** in
  focused groups (two expanded motion/full-body journeys + six existing regressions). Covers
  layout/face preservation, initial differential-mask contract, pose targets, API validation,
  concurrent rename, drawn-emote immutability, real rendered blink halves/replacement,
  grouped/exclusive controls, settings reload, rooted motion, exports and fit-guide workflow.
  Final unit suite: 90 passed in 44.14 s. Physical webcam feel and human Cubism import remain open.
- Review files (ignored): `outputs/fidelity-review/` has logs, superseded trials, screenshots and
  `natural-sway-sparkles.webm` (~0.85 MB, VP9, 1356×1216 incl. transparent margins, last video
  packet at 2.659 s with the 3 s recording setting). Browser export stalled the tool for a long
  interval; no extra recording was attempted. Device controls were off during final inspection.
  Final screenshot: `final-full-body-preview.png`; original-model emote screenshot:
  `sparkle-aura-preview.png`. Final browser reports no application errors. See
  `docs/OUTPUT_QUALITY.md` for the prioritised further-quality list and remaining limits.
- Final server restarted only after the queue was idle, reloader off, real mode at
  **127.0.0.1:5055**, launcher PID **11212**, Python child **9148**; logs
  `outputs/fidelity-review/server-final.*.log`. No commit or push.

Stance guides, automatic checks, kept head features, refinement, visemes, alternate looks and motion (2026-10-23, late, uncommitted):

- Owner (pasting ChatGPT's list): draggable shoulder/hip/ankle guides; automatic checks for
  duplicate faces, cropped shoes and uneven shoulders; preserve original head layers when
  extending; targeted hand/face refinement and painted A/I/U/E/O mouths; alternate hairstyles and
  outfits with generated hidden regions; "watch gifs and videos … to truly nail down vtuber
  movement"; more creative toggles.
- **Stance:** Studio fit preview dots for shoulders (OpenPose 2/5), hips (8/11), ankles (10/13);
  each pair mirrors about the axis. `outpaint.check_guides` validates `guides` ({name: [half
  width, y]}, top to bottom) on `/extend` and `/extend/<id>/plan?guides=`; `pose_points` derives
  elbows/wrists/knees from them. `_character_mask` is lru-cached per base image mtime (dragging
  re-plans). Reset stance button.
- **Checks** (`src/core/checks.py`, run after full-body extends, `metadata["checks"]`): second
  face = WD tagger on `face_tiles` down the body (the whole picture stays "solo" ≥0.95 even with
  a second head; tiles with a face score eye colour 0.70-0.89, body tiles ≤0.06); feet/top at
  the border; `shoulder_tilt` (hair-coloured pixels removed, first row reaching 0.4 head from the
  axis per side, >0.1 head = uneven); tagger "bad hands". `tagger.tag_images` loads once,
  download=False inside jobs. Studio `#resultChecks` with fixes: Try another seed (re-extends
  the source with the saved options; `info["outfit"]` keeps the outfit words), Make room
  (uncrop), Repaint this area (ellipse mask → /edit), Refine hands. Real data: 0 false alarms on
  7 full bodies; 3/4 superseded alignment trials flagged (shoulders); 6/6 synthetic second heads.
- **Kept head features** (`src/core/heads.py`, after separation when `request.extension` names
  an existing source): source eyewhite/irides/eyelash/eyebrow/nose/mouth layers scaled into
  `placed` replace same-named new layers where ≤8 % of their pixels differ from the retained
  source picture; painted expressions copy when eyes+mouth were kept, and `_paint_expressions(keep=True)`
  then skips repainting (no SDXL load). Face/hair/ears are NOT swapped: on the real hoodie
  (`c2e836b7` copy) that left a grey forehead smudge and reordered ears vs earrings. Kept
  features were visibly cleaner (complete lashes, eyebrows). `source_model` must match `[\w-]+`.
- **Refine** (`src/core/refine.py`, `POST /refine/<id>` {target hands|face}, operation
  "refine", `request.regions`): boxes from handwear-l/-r (lowest 0.6 face widths) or the saved
  pose (recomputed for older plans), face from `anatomy`; each box inpainted alone with
  `context=0.15`, part-only prompts. GPU: hoodie `a834d4e3` hands → **`8b761f2f`** (new sketch):
  defined fingers, smears gone, no seam (~30 s).
- **Visemes:** `expressions.VISEMES` / `paint_visemes` (`mouth_a … mouth_o`, `request.visemes`;
  Studio/preview chip id `visemes`). Preview PSD **v16** inserts them after the mouths; rig slot
  `fade:'viseme'`; they replace mouth_open by weight; `RigVisemes.stackAlpha` makes the stack
  composite to the weighted mix. `static/vendor/anime25d/lib/visemes.js`: LPC (order 12 at 12 kHz,
  Durand-Kerner roots, bandwidth < 500 Hz) → F1/F2, pitch (shortest lag within 85 % of the best
  autocorrelation peak) scales male references by (f0/120)^0.25, F1 counts half. Synthetic
  vowels 30/30 nominal, ≥85 % with ±8 % formant shifts (Node test). Mic uses a second 2048-pt
  analyser; Auto talk picks vowels by Japanese frequency; camera maps mouth width/roundness.
  GPU on witch `004b7e89`: A/I/U/E/O all distinct and usable. Known: the closed↔open fade is
  pale for ~50 ms (keeping the closed mouth solid showed its line through small O/U: reverted).
- **Toggles added:** drawn Question mark, Exclamation, Zzz (region "above"), Music notes,
  Floating hearts (aura box; float with Idle via `FLOATING`), drawn **Cat mouth** (ω in the
  mouth's ink over sampled skin; painted ":3" erased the mouth), painted Tongue out and Fang.
  Mouth emotes replace the talking mouth (`MOUTH_EMOTES`, exclusive `variantPart:'mouth'`).
  Steam puffs were tried and dropped (read as earmuffs).
- **Alternate looks** (`src/core/variants.py`, `POST /variant/<id>` {part hair|top|bottom|headwear,
  prompt}, job operation "expressions" with `request.variant`; stored in
  `expressions.json["variants"]`, preview layer `toggle Hair: …` after the part's top layer;
  rig `VARIANT_SLOTS`/`replaces`, one per part). Graded inpaint of the part's visible area over
  the character without it; keep = changed ∩ character ∩ rules. Six GPU iterations on the witch
  (each a lesson, now comments in code): brief hat tags drew a second hat (dropped, "hat" negative);
  braids became a cape (body/ears/hat keep their layers; bare canvas pixels must be nearer the new
  hair colour than the canvas); grown bangs ghosted lashes (face pixels only inside old bangs,
  feature guard 4 % face width); See-through files part of a hat in back hair (back-hair pixels
  above the face stay as hat, carried inside the variant). Final `hair-1` "short black bob" on
  the witch: clean bob, live face/ears/dress; remaining: a strip of old silver hair under the
  brim and faint marks of See-through's under-bangs forehead. Longer-than-original hair cannot
  grow over the body (by design). No delete button for a variant yet.
- **Motion** (studied Live2D's open sample motions/framework; numbers in `docs/MOTION_NOTES.md`):
  `Motion.blink` (0.09 s ease-in close, 0.04 hold, 0.18 ease-out open), `Motion.idleHead`
  (Live2D's 6.5345/3.5345/5.5345 s periods at ~60 %), `TURN_TILT` 0.3 (non-camera), eyes lead
  random head moves by 130 ms, idle saccades, dip on fast turns + blink on big turns, speech
  beats. Browser: idle varies, blinks, no console errors.
- **Hardening:** `expressions.read` uses utf-8-sig. A PowerShell `Set-Content -Encoding utf8`
  rewrite (BOM) of the witch's expressions.json nearly made a running job read {} and overwrite
  16 toggles; caught and rewritten in time. Never edit these JSON files with PowerShell 5.1.
- Verified: 98 GPU-free pytest + 8 Edge e2e (full-body journey now drags a hip dot and checks
  `#resultChecks`), both Node suites. Server restarted only with an idle queue; final PID in
  `logs/server-5055.*.log`. The witch's toggles/visemes/variant and the refined sketch
  `8b761f2f` are real test outputs; owner's preview settings were not saved. No commit or push.

Neck seam and head/body size (2026-10-24, committed as `7bdb230` with all earlier passes and pushed to main):

- Owner, looking at Sol's alignment models: "the weird seam in the neck feels off … ensure heads
  and bodies correspond in size". Superseded trials 8f290a89/eab60823/24371d1a/2d168bcc and
  4a46a1a0 had been deleted by the owner; compared on `a834d4e3`, `c2e836b7`, `42991481`.
- Seam cause: the plan kept original pixels to chin + 0.1 head and ramped to full repaint by
  chin + 0.3 head, so the source's neck shading (hoodie: a darker plate ending in a line across
  the collarbones, lavender scraps) and collar (witch: a high collar cut through its gem above a
  strappy top) survived. Now the ramp starts at the jaw and is full by chin + 0.15 head.
- Size cause: the figure was scaled by crown-to-chin head height, which includes hair volume
  (hoodie 1.53 face widths vs 1.33 measured on this checkpoint's own full bodies `4a66c60e`,
  `dd6e1715`), and shoulder joints sat 1.16 head heights apart. Measured in face-based heads,
  Sol's extensions were 8.0-8.4 heads tall (natural figures 6.3-6.7): small heads on broad
  bodies. Now `outpaint.body_unit` = 1.33 x face width sizes the body below the chin
  (`result_geometry["unit"]`), pose shoulders 0.53 unit (~0.7 face width) each side; older
  plans without "unit" keep the head-height behaviour. `checks.heads_tall` flags >15 % off the
  target ("head looks too big/small", fix: another seed).
- GPU, same seeds/options as Sol: hoodie → **`c05b46df`** "Hoodie - full body v2 (clean neck,
  head size)" 7.1 heads (was 8.0), clean neck and newly drawn choker; witch → **`af5b9c67`**
  "Witch - full body v2 …" 7.1 heads (was 8.1), the dress's own halter neckline. Both all clear.

## 11. How to work in this repo

- Smallest complete change; reuse existing code and libraries; no speculative abstractions.
- Every non-trivial logic change gets a small GPU-free pytest (monkeypatch GPU parts like
  `tests/test_pipeline.py` does). Demo mode must keep working without torch/GPU.
- For anything touching the GPU path, verify on the real GPU once, watching `nvidia-smi` for
  spills. Never run two GPU jobs at once; kill stray `python.exe` processes you started.
- Verify UI changes in a real browser, not just by reading code.
- Report honestly: what you verified, what you didn't, and measured numbers.
