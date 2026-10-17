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
- Branch: `claude/automated-vtuber-model-011CUg6gEfYhoWvMuuXdF2hJ` (the real development branch;
  the default branch `Ka-myii` is stale; there is no `main`).
- **Everything is committed** (2026-10-17). The owner asked to make Ka-myii a good initial
  product to push to GitHub; M1–M3 were committed as one baseline (`c113da1`) and M4 work
  (section 9) as separate commits on top. Commit or push only when the owner asks.
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
| Tests (no GPU, no torch needed) | `.\.venv\Scripts\python.exe -m pytest tests -q` → 35 must pass; on sandbox permission conflicts add `-p no:cacheprovider --basetemp <fresh dir>`. Plus `node tests/preview_runtime.test.cjs` and `node tests/tagcomplete.test.cjs`. CI (`.github/workflows/tests.yml`) runs the same on Windows + Ubuntu without torch. `tests/conftest.py` forces `seethrough.available() == False` so no test can start the real engine. |
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
- **Failures** with artwork are saved as `failed` and shown in the Gallery with the error and a
  **Retry layers** button (`POST /api/generation/retry/<id>` re-queues the artwork as an upload).
- Studio: LoRA weight slider (saved LoRA + weight restored on reopen), a notice when See-through
  isn't installed (`/model-status` → `seethrough`), reopened results select the checkpoint used.
- README rewritten with screenshots (`docs/images/`).

Verification (2026-10-17, owner's laptop):

- 35 pytest (also in a fresh torch-free venv) + 2 node tests pass.
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

## 10. How to work in this repo

- Smallest complete change; reuse existing code and libraries; no speculative abstractions.
- Every non-trivial logic change gets a small GPU-free pytest (monkeypatch GPU parts like
  `tests/test_pipeline.py` does). Demo mode must keep working without torch/GPU.
- For anything touching the GPU path, verify on the real GPU once, watching `nvidia-smi` for
  spills. Never run two GPU jobs at once; kill stray `python.exe` processes you started.
- Verify UI changes in a real browser, not just by reading code.
- Report honestly: what you verified, what you didn't, and measured numbers.
