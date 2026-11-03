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
  (2026-10-28: the optional 3D avatar path runs Pixal3D in a **private, pinned ComfyUI that
  Ka-myii's own installer puts in `engines/comfy3d`** — the same pattern as See-through's engine,
  never the user's ComfyUI. The owner asked for 2D or 3D avatars from the Studio and allowed taking
  components from Atelier3D, which uses exactly this engine. Model files are hard-linked from a
  sibling Atelier3D checkout when present.)
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
- **Everything below is committed** and pushed to `main` and the development branch: M1–M5
  (2026-10-17), the 2026-10-18…24 passes (2026-10-24), and on 2026-11-03, at the owner's request
  ("push to main … when you are satisfied"), the 3D VRM avatars, the QoL passes, chest physics
  (Claude + Codex) and moods/face marks/animal ears in one commit. Commit or push only when the owner asks.
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
| Tests (no GPU, no torch needed) | `.\.venv\Scripts\python.exe -m pytest tests --ignore=tests/e2e -q` → 164 must pass; on sandbox permission conflicts add `-p no:cacheprovider --basetemp <fresh dir>`. Plus `node tests/preview_runtime.test.cjs`, `node tests/tagcomplete.test.cjs`, `node tests/3d_tracking.test.cjs`, and `node tests/3d_motion.test.cjs`. CI (`.github/workflows/tests.yml`) runs the GPU-free suite on Windows + Ubuntu without torch. `tests/conftest.py` forces `seethrough.available() == False` so no test can start the real engine (likewise `mesh3d.available()` and DWPose's `pose.model_path`). |
| 3D engine (optional) | `.\.venv\Scripts\python.exe scripts\install_3d.py` → `engines\comfy3d\` (ComfyUI v0.42.1 + own venv, torch 2.14.1+cu130 via uv hard links, Pixal3D/MoGe/DINOv3/VAE files hard-linked from `..\Atelier3D\models`, DWPose into the HF cache). Engine log per job: `outputs\<id>\avatar3d\engine.log`; port 8198 (`KAMYII_3D_PORT`). |
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
src/core/mesh3d.py             3D mesh: frames the cut-out, Atelier3D's Pixal3D graph (no BiRefNet), starts/stops the
                               private ComfyUI per job (HTTP polling, progress from its KSampler tqdm lines)
scripts/comfy3d_nodes/kamyii3d engine node copied in at every start: Atelier3D's photo paint + a JSON camera-fit report
src/core/pose.py               DWPose ONNX (top-down on the mask's box, no YOLOX): 133 whole-body keypoints
src/core/rig3d.py              GLB read, camera <-> artwork mapping, joints, skin weights, hair springs, T-pose,
                               expression decals, VRM 0.x writer, demo mannequin
AssemblyLine.make_avatar3d     job op "avatar3d" (Gallery) or after a "generate" with request.avatar3d (Studio 2D + 3D);
                               builds in avatar3d.new/, swaps into avatar3d/ only on success
tests/test_avatar3d.py         GPU-free 3D tests; tests/e2e/test_3d_stage.py Gallery -> Make 3D avatar -> stage journey
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
   2026-11-01: no CORS still let other sites send *simple* cross-site POSTs (an auto-submitted
   multipart upload form would queue a GPU job). `reject_cross_site_writes` in app.py refuses
   non-GET requests labelled `Sec-Fetch-Site: cross-site/same-site` (same-site = another
   localhost port) or with an `Origin` that isn't this host; test clients without the headers pass.

21. **Cubism files are checked against Live2D's official JSON schemas** (embedded in the
   markdown of github.com/Live2D/CubismSpecs, `FileFormats/*.json.md`; that repo has no license,
   so they are not vendored). model3/cdi3/motion3 passed; physics3 lacked required Meta fields
   and drove undeclared, invalid ids — fixed. `test_cubism_files_follow_the_specs_…` checks the
   rules without the schemas; re-validate with them after changing the assembler.

22. **3D: everything lines up through Pixal3D's camera.** The mesh is glTF-framed (Y up, faces +Z,
   the character's left is +X); the input crop was seen from (0, 0, f), f = 0.5 / tan(fov/2), fov =
   MoGe's, reported by a `PreviewAny` node. Ka-myii frames the crop itself (`mesh3d.frame`, the math
   of ComfyUI's ImageCropToMask, pad 1.1, subject on black) instead of BiRefNet, so crop <-> artwork is
   exact. The photo-paint node's silhouette fit (moments, score, azimuth, flip) is returned as JSON and
   stored in `camera.json`; `rig3d.Camera` refuses to rig below 0.85 match or a turned-around mesh.
   Bard run: match 0.98, 19% of texels painted from the artwork.
23. **Rig joints:** DWPose gets the shoulders/elbows/wrists/hips/knees/ankles/feet/68 face points/hands
   right on clean anime full bodies (bard 0.65–0.9 scores; RGB input, mmpose means). But its elbow sits
   on the arm's *outline*: the ray there grazed the arm and hit the cape (elbow 18 cm too deep, ray
   "thickness" 1 mm). Limb joints therefore take the front-most hit among 9 nearby rays minus a radius
   from anime proportions (x shoulder-keypoint width); torso joints use the middle of the first solid.
24. **Torso weights by height, not nearest segment**: the neck segment is tiny, and 37% of face vertices
   went to the neck (face tore on head turns). Blend zones at joints are absolute (25% of the shorter
   neighbour). Head pivot at 80% from the shoulder line to the chin; head height = 2.2x eye-to-chin.
25. **Arms: label the surface, then cut fused contacts.** Capsules alone took the cape/thigh/pouch and a
   too-soft falloff left the arm's own surface 66% arm (T-pose streaks). Now: front-surface vertices take
   the top See-through layer at their pixel (`handwear-l/-r` are exactly the bare arms + hands), labels
   spread over the surface by Dijkstra (`spread_parts`, ~0.7 s for 66k vertices), an arm = its own part
   within 2.5x radius. Pixal3D fuses hands to thighs/pouches; triangles joining arm and body more than
   half an upper arm from the shoulder are deleted (~500 of 100k on the bard) — 0 stretched edges after;
   `cap_holes` then fans each opened loop around its centre with that side's weights (closed mesh again).
26. **VRM 0.x needs a T-pose rest** (apps lower the arms from it; an A-pose rest would be pushed into the
   body). The A-pose arms are raised by LBS with the final weights; three-vrm/stage `relaxArms` lowers
   them again (round trip looks like the artwork). VRM 0.x faces -Z: export flips x and z.
27. **Expression decals**: closed eyes / open mouth from `expressions/` become copies of the face
   triangles under them (front-facing, front-most), UV'd into an atlas of the painted crops, 2 mm under
   the skin; a morph target pushes them 2 mm out. All of them share one primitive because glTF requires
   every primitive of a mesh to have the same morph targets. A drawn-open mouth uses `mouth_close`
   shown by default and pushed in for "A".
28. **Engine polling**: `/history/<id>` timed out (5 s) while a node held the GIL; poll with 30 s and
   retry. `MoGe inference:` prints its own tqdm bar — only unnamed bars count as sampler steps.
   Real numbers (2026-10-28, Standard, 100k faces, 2K): engine start ~25–50 s, prompt 178 s, rig 10–13 s;
   **peak VRAM 6.99 GB** of 8 GB through the app (no other GPU users) — close to the spill point.
29. Editing this repo through bash heredocs: `\n` inside a heredoc'd Python string became a real newline
   and broke `scripts/install_3d.py` once (Atelier3D noted the same); patch with Edit or a script file.
30. **Generated VRMs pass the official VRM 0.0 JSON schemas** (github.com/vrm-c/vrm-specification,
   `specification/0.0/schema/*.json`, Draft 4; not vendored): 0 errors on the bard avatar, and the check
   flags planted bad `allowedUserName` / bone / preset values. Re-check after changing `write_vrm`:
   download the 15 schema files, then `uv run --no-project --with jsonschema --with referencing` a
   Draft4Validator over `extensions.VRM` with a `referencing.Registry` of the files.
   Khronos' glTF validator (npm `gltf-validator` 2.0.0-dev.3.10, `validateBytes`): 0 errors, only the
   inherent VRM 0.x warning `INVALID_EXTENSION_NAME_FORMAT` ("VRM"); unused joint slots point at joint 0
   and skinned mesh nodes sit at the scene root because it flagged both.
32. **Pixal3D's GLB is split along its UV seams** (bard: 70k vertices, 50k positions, 38k open edges).
   Label spreading, weight smoothing and hole capping all run on `weld(pos)[faces]` (first vertex per
   position) and results are copied back to every copy. Before this, a third of the mesh got no part
   label and the T-posed arms came apart chart by chart; seams could also crack under posing.
33. **3D closed-eye decals used to leave the open iris showing** (witch: ~16% of iris pixels had
   no decal triangle; Joy/blink looked half-open). Two causes: (a) decal triangles needed *all three*
   corners to be front-most, and Pixal3D triangles are big (~14 px), so one corner touching a bang
   strand that crosses the eye dropped a whole chunk; vertex normals are also noisy there (p99 57°
   from the face normal). Now: facing by the triangle's own normal and 2 of 3 corners front-most
   (iris coverage 83.8% → 99.8%, no triangle more than 5 cm behind picked up). (b) the painted
   closed eyes are feathered for the 2D rig; `cover_open_eyes` makes the blink and > < / spiral
   decals opaque over the eye-white/iris/lash area (closed eyes on the face layer's inpainted skin,
   the artwork's own hair where bangs cross). Two tiny specks remain on the witch (probably iris
   colour painted onto strand geometry). Debug trick: a solid magenta eye_close shows coverage.
31. **Expression toggles in 3D**: painted toggles whose region is on the face (cheeks, tears, eyes,
   temple, forehead, mouth) become custom VRM expressions (`presetName: "unknown"`, named by the
   toggle's label, e.g. "Blush"); floating symbols (above/aura/notes/hearts/gloom) are skipped (no
   surface). Painted A/I/U/E/O mouths replace the generic open mouth. Each decal sits a little
   further out than the one before (2 mm + 0.5 mm each) so two shown at once never z-fight.
34. **Chest springs in 3D** (`rig3d.bust_springs`, rig 3): per side a non-humanoid `bustL`/`bustR`
   bone under `chest` (root 30% of the way from the chest joint to the front surface, child `…_tip` *at*
   the surface, so the spring's tail points forward and vertical body motion pitches it = up/down
   bounce). Weights: a Gaussian (σ = 0.16 × shoulder width) at ±0.22 sw, chest height, × front ramp ×
   the vertex's chest+spine weight × `parts == "body"` (arms, hair, neck never move). Own VRM boneGroup
   "bust" (stiffness 1.2, drag 0.2, no gravity). **Skipped when the mesh ends above the under-bust**
   (witch bust: the cut edge would wobble). The tip needs an explicit node: three-vrm/VRM 0.x give a
   leaf joint a 7 cm virtual tail along its offset from the parent, which for this bone points sideways.
35. **Drawn face marks in 3D** (`rig3d.DRAWN_MARKS`, rig 4): `decals()` adds `toggle:blush/gloom/anger/sweat`
   for every avatar whose character has no painted one, drawn on the CPU from the 2D layers
   (`expressions.draw_blush` for cheeks, `draw_symbol` for the rest, regions from `toggle_mask`).
   Gloom is cut to skin the front hair doesn't cover: in 2D it sits *under* the bangs, but a mesh
   decal lands on the front-most surface, i.e. on the bangs, with jagged triangle edges. The marks
   need `assets/` layers with eyes and a face; without them (tests' bare fixtures, imported files)
   nothing is added. Sorrow binds tears **and** gloom.

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

Redraw facing forward (2026-10-24, uncommitted):

- Owner (Gallery screenshot: an imported bard `72e7b3cf`, side-on, holding a harp → the straight-on
  witch/hoodie sketches → the front-facing bust): "add the option to turn character submissions in
  dynamic poses to be redone in the VTuber forward looking style".
- `/extend` mode **`front`**: `outpaint.plan` keeps nothing (mask 255, strength 1.0, no `placed`, so
  no head-layer transfer) and always uses the standing pose guide at its default centred figure
  (chosen proportions). The brief loses pose/view/prop tags (`POSE_TAGS`: holding, instrument,
  from side, looking to the side, sitting, …) and gains `FRONT_TAGS`; `FRONT_NEGATIVE` forbids
  profile/holding. Identity therefore comes from the tag brief only: colours/outfit survive,
  small details (patterns, accessories the tagger missed) can change. Checks run (duplicate,
  feet/top, hands); head-size/shoulder checks need result_geometry, which a front redraw lacks.
- UI: Studio "Complete the body" checkbox **Redraw facing forward** (disables keep-face/guided,
  hides the fit guide); Gallery card More options → **Redraw facing forward** opens
  `/generator?model=<id>#front`. "Try another seed" re-sends the same mode.
- Verified: 99 GPU-free pytest, new Edge journey (gallery link → checkbox → artwork sketch with
  mode front) + the full-body/import/export journeys. **Not run on the GPU yet** (the owner's own
  server on :5000 held 4 GB RAM / the GPU with 1.5 GB free; it needs a restart for the new code).
  Not done: IP-Adapter/reference identity (new ~3 GB download), an upper-body front pose,
  offering it in the upload form.

Clothing-faithful front redraws (2026-10-25–26, uncommitted):

- Owner compared imported bard **72e7b3cf-de5c-4c98-9564-9d14f37f90c1** (Study 9C9A067E)
  with front redraw **7b5fa8a3-46e6-45d0-b516-8abe5c627b61** (Study C9E483B4): long gold
  dress/cream underskirt and embroidered cloak had become a short, simplified outfit.
  Cause: all-white front mask + strength 1 meant the reference pixels contributed nothing;
  WD tags missed garment length, colour and trim. This supersedes the text-only limitation above.
- **Visual reference:** new lazy `src/core/reference.py`, IP-Adapter Plus SDXL ViT-H from
  `h94/IP-Adapter`, Apache-2.0, pinned **018e402774aeeddd60609b4ecdb7e298259dc729**.
  About 3.4 GB of encoder/adapter weights in HF cache, no package upgrade or ComfyUI dependency.
  Fit the entire image into a white 224px square before CLIP; default centre cropping would
  discard hems/boots. Source pixels are never edited; all inference is local.
- `/extend` mode front accepts **reference_strength 0..1**, default **.65**, saved in extension.
  Zero bypasses the visual encoder/adapter and keeps text-only behaviour. The staged input
  image itself supplies the reference: no new client-controlled filesystem path.
- **Memory order:** text embeddings → release both text encoders → encode image with ViT-H →
  free image encoder → load adapter into UNet → existing FP8-storage/FP16-compute casting →
  ControlNet. Negative/positive patch embeddings passed explicitly (CFG ordering tested).
  Existing .86 CUDA allocator cap retained; temporary adapted pipeline unloaded after work.
  Preparation errors now share the cleanup finally, including allocator-cap restoration.
- **Clothing prompts:** removed mandatory `legs, feet, shoes` / `long legs` additions which
  competed with longer clothing; standing guidance still controls anatomy. Pose filters match
  tag starts, preserving e.g. `dress with crossed lacing`. Redraws now honour saved steps/CFG
  (the repaint caller previously dropped them).
- **Studio:** Reference influence slider, download/pose tradeoff explanation, detailed outfit
  placeholder. Setting, outfit and proportions restore on reopen; quality-check retries use
  the original source and saved reference strength. **Compare with the reference** shows source
  and result side-by-side, with an original-artwork link; source ID validated before URL use.
- **Real GPU:** first .65 trial, seed1229928384, Animagine XL4, 28 steps/CFG5, 329.7s incl.
  cold imports/loading, `outputs/redraw-review/reference-trial.png`. Long gold dress, cream
  underskirt/cuffs, blue/gold cloak, braid and pouch survived; clasp/trim/style still changed.
  Final complete API job **gen_daddbc4d**, .70 + explicit outfit details, same seed, saved CFG6.5,
  created **e49689a3-0d2c-4738-bd2b-8470332e73dc**, **Bard - faithful outfit redraw**.
  **357s**, sketch for review (Separate layers available), automatic checks all clear.
  Both measured **4760MiB allocated / 5034MiB reserved**, GPU0MiB after completion.
  The second load slowed under severe system RAM pressure (~173MB free); stopped only our
  concurrent Edge/test processes, then generation finished. Do GPU and browser QA sequentially
  on this laptop. No owner generation was cancelled; only the idle app server was restarted.
- **Limitations:** visual guidance improves resemblance, not exact garment reconstruction;
  embroidery/fasteners, hair silhouette and drawing style may change. Higher strengths can
  copy the unwanted pose/props. Tested Animagine XL4; other SDXL/LoRA combinations unverified.
  Full-body extension and generic region edits retain their existing conditioning paths.
- **Verification:** 114 GPU-free tests pass across runs (113 in the suite, one existing export
  test's sandbox-denied directory rename passed on an escalated retry); both Node suites pass.
  New tests cover encoder cleanup on success/cancel/failure, full-image fit, CFG ordering,
  memory sequencing, zero bypass, invalid strength values and source immutability.
  Final expanded Edge journey passed (78.30s): saved strength/outfit, comparison, phone layout,
  and retry using original source/settings. Real-result comparison inspected, no page errors;
  screenshot `outputs/redraw-review/final-comparison.png`. No commit or push.
- Final real server on127.0.0.1:5055, reloader off, launcher **32736**, logs
  `outputs/redraw-review/server-final.*.log`. Originals and prior uncommitted front-redraw work preserved.

3D avatar stage foundation (2026-10-27, uncommitted):

- Owner wants to create real 3D VTuber characters with tracking, like professional VTubers.
  This is a new, long-term product goal. A true avatar needs 3D geometry, a skinned humanoid
  rig, facial expressions, materials, spring bones and a compatible export. Existing PSD/2.5D
  layers do not provide those automatically. Do not present the stage as 3D generation.
- **First milestone:** `/3d`, linked in the main navigation, opens a local `.vrm` avatar or a
  static `.glb` file in the browser (`templates/3d_studio.html`, `static/js/3d_studio.js`).
  Orbit/zoom, fit, head turn/tilt, mouth, idle, blinks, spring-bone update and transparent
  PNG are supported when the avatar supplies those features. T-pose upper arms are gently
  lowered; already lowered arms are left as-is. VRM missing blink/mouth expressions is reported;
  unrigged GLB has face controls disabled. File bytes remain in the browser. Invalid files and
  glTF references to external resources are rejected; failed replacements preserve the model.
- `static/js/3d_tracking.js` bridges existing local FaceMesh and `FaceFeatures.Tracker` to
  compatible VRMs: head, independent blink and mouth, optional neutral calibration, a 1.2s
  dropout hold, and cancellation/permission/disconnect cleanup. Camera off by default; no face
  frames leave the device. This is face/estimated head tracking, not full-body/hand mocap.
- Vendored **three.js 0.180.0** and **three-vrm 3.5.5**, MIT, in `static/vendor/vrm3d`;
  licenses included, credited in README. No CDN, new GPU packages or new server upload API.
  VRM0 orientation is corrected with `VRMUtils.rotateVRM0`; each avatar has its own license.
- **Real browser test:** VRoid beta sample Vivi (CC0; official license
  https://vroid.pixiv.help/hc/en-us/articles/360014900273) downloaded only into ignored
  `outputs/vrm-review/Vivi.vrm`, SHA256
  `EAF902E041A7A810F1423599AE75682F61184AB6EF0206DA9A8ED9CAA8EC3A9D`.
  VRM0 avatar renders, relaxed arms, pose/mouth controls, reset and local assets work; screenshot
  `outputs/vrm-review/vrm-stage-final.png`. A tiny synthetic GLB checks static rendering,
  rejection of external links/corrupt inputs, preservation after failed replacement and 390px
  width. `tests/e2e/test_3d_stage.py` real VRM test needs `KAMYII_TEST_VRM` pointing to Vivi;
  otherwise it skips. VRM1 and physical camera tracking remain untested.
- Verification: **114 GPU-free pytest**, both existing Node suites and
  `node tests/3d_tracking.test.cjs` passed. Edge VRM/GLB journeys passed; final camera denial
  check with real Vivi and mocked camera denial passed twice (retry works, stays off).
  3D files, camera and PNG stay on-device. No owner projects modified.
- **Generation research:** `docs/3D_ROADMAP.md` records a staged local plan. TripoSR is MIT and
  upstream reports ~6GB VRAM for a raw unrigged mesh, but its Windows/native setup, quality and
  16GB RAM fit are unverified. Free physical RAM was ~0.5–1.6GB, so no 3D generation attempt,
  installation or model download was started. This machine has CUDA nvcc 11.8; current
  torchmcubes needs CUDA 12+ for its GPU build and native build tools are not on PATH. UniRig's
  >=8GB/native deps and Windows fit also need testing. Do not treat a raw GLB as a VRM rig or
  guarantee professional-quality 3D from one image. Close unnecessary apps only with owner choice.
- The real server was started with empty queue at 127.0.0.1:5055, launcher PID 11852, reloader
  disabled; logs `outputs/vrm-review/server.*.log`. No commit or push. Existing uncommitted work
  remains intact. Physical webcam/mic and human Cubism import remain outstanding.

3D avatar generation (2026-10-28, Claude, uncommitted):

- Owner: improve 3D model generation "so the user has the option to create 3d or 2d avatars in the
  studio", taking Atelier3D components as needed. Built roadmap stages 2–4 in one path:
  artwork -> Pixal3D mesh (mesh3d.py, Atelier3D's graph + photo-paint node) -> DWPose joints, skin
  weights, hair spring chains, T-pose, painted blink/mouth decals -> **VRM 0.x** (rig3d.py).
- UI: Studio **Avatar: 2D / 2D + 3D** (3D locks Full body and adds "standing, arms at sides, facing
  viewer"; the 2D character is kept if 3D fails, with the reason in `metadata.avatar3d_error` and a
  note under the result); result button **3D avatar ↗**; Gallery **Make 3D avatar** / **3D avatar ↗**
  / More options **Download VRM**, **Rebuild 3D avatar**; 3D stage picker of Ka-myii avatars,
  `/3d?model=<id>`, **Download VRM**. API: `POST /api/generation/avatar3d/<id>` {quality},
  `/generate` {"avatar3d": true|{quality}}, artifacts `vrm` and `mesh`, `/options.avatar3d`.
  Job stages `avatar3d` / `generate3d` in jobs.py. Demo mode builds a box mannequin VRM.
- Installer `scripts/install_3d.py` (run.bat/run.sh offer it, default No, `engines/comfy3d.skip`).
  Installed on this PC: 0 extra GB for models (hard links), venv from the uv cache.
- **Verified:** real GPU runs on the bard front redraw **7b5fa8a3** (scratch run, then a real API job
  `gen_a77cab36` on a .venv server at :5058: ~4 min, peak 6.99 GB VRAM). Its avatar is in
  `outputs/7b5fa8a3-…/avatar3d/` (re-rigged afterwards with hair springs: 2 chains). Renders in
  three-vrm (headless Edge): relaxed pose looks like the artwork, head turn/tilt clean, wave pose,
  her own blink and open mouth; stage page loads it with no page errors. VRM 0.0 schema: 0 errors.
  Final checks: 128 GPU-free tests (14 in `tests/test_avatar3d.py`), Node suites, all 9 Studio/Gallery
  journeys, all 4 3D-stage journeys (incl. Vivi + expression chips, Gallery -> stage, Studio 2D + 3D).
- **Studio 2D + 3D, real GPU** (job `gen_10f12ddc`, brief "1girl, pink hair, long hair, twintails, ...",
  novaAnimeXL): the 3D tags gave a full-body, front, arms-down figure on white; 2D finished (SDXL,
  See-through, expressions; 1247 s in all, slow because ~1.5 GB RAM was free), peak VRAM 7.4 GB over
  the job. Its 3D stage **failed** (`hair_springs`: a strand starting a link below the jaw had an empty
  first band -> IndexError); the 2D character was kept with `avatar3d_error`, as designed. Fixed (nearest
  hair vertices by height, regression test) and springs made non-fatal. Character **72213aa1**.
- **After the fixes (welded topology, caps, outlines), real GPU, through the API on :5058:** idol
  **72213aa1** (Gallery route, `gen_48d8457f`, ~4 min, peak 7.06 GB): clean T-pose with whole arms,
  relaxed pose matches the art, 4 hair chains (twin tails), her own ^^ blink and open mouth. Witch
  bust **004b7e89** (upper body, `gen_005ba176`, peak 6.58 GB): 0.5 m bust, painted A I U E O, 11
  face toggles as VRM expressions (Heart eyes + Blush, > < eyes + Tongue out checked), braids on 6-7
  spring chains swinging in the live stage. Both re-rigged on the CPU with the final code (`rig3d.build`
  on the existing `avatar3d/`). README image `docs/images/avatar3d.jpg` comes from these renders.
- Hard case: the bard's cape is drawn over her upper arms, so the upper arm reads as cape and only the
  forearm rises in the T-pose (relaxed poses look right). Tried letting the inpainted `handwear` layer
  claim covered upper-arm vertices: +16 vertices, no visible change (Pixal3D fused the upper arm into
  the cape), so it was removed. A real fix needs a separate arm in the mesh (multi-view input).
- **Not verified:** VSeeFace / Unity / VRoid Hub import by a human; characters other than the bard
  (busts, long skirts, held props); spring-bone behaviour beyond a quick turn; physical webcam.
- Body MToon has a 0.06 cm world-space outline (`OUTLINE_CM`), decals none.
- **Known artifacts:** flat caps/slivers where cut fused triangles were (cape by the upper arms),
  thin arms in the T-pose (Pixal3D builds the arm as a relief on the cape), backs are Pixal3D's guess.
- Ideas next: cut-and-cap fused contacts instead of deleting; finger bones from DWPose hands;
  eye bones/look-at; multi-view (front + back Redraw) Pixal3D for better backs; Draft/High in the UI.

3D rig, expressions and reliability (2026-10-28, uncommitted):

- Owner asked Codex to continue Claude's 3D pipeline and make it stronger. They explicitly chose
  **continue without a GPU test**; no 3D engine/GPU run was started. Earlier Claude Pixal3D
  results are evidence for the old engine path only. The new engine startup/polling changes below
  still need an end-to-end GPU run on the owner's 8GB laptop before claiming them verified.
- **Rig quality:** actual hair is separated from rigid head attachments in `rig3d.py`; hats,
  human/cat ears and earrings follow the head instead of acquiring hair spring chains. Vertices
  projected outside source-image bounds no longer borrow border labels or occlude valid vertices.
  Focused synthetic tests cover both. Existing witch/idol/bard meshes were re-rigged as copies in
  `outputs/vrm-review/rig-quality/`: witch 99,982 triangles/6 hair chains/16 painted toggles and
  visemes; idol 99,746/4/3; bard 99,801/3/3. Counts and expression sets match their baselines;
  original files are unchanged. No `write_vrm` schema was changed.
- **Saved mesh refresh:** Gallery More options → **Refresh rig & expressions** submits
  `/avatar3d/<id>` with `reuse_mesh: true`; it runs DWPose and `rig3d.build` on the CPU using
  existing `mesh.glb`, `camera.json`, `crop.png`, with no Pixal3D launch. Requires all three files,
  reports when unavailable, retains original mesh/quality/seed and carries current painted face
  expressions. API rejects this option for a new text-to-character request. Gallery list adds
  `can_refresh_rig` in the response only when the required files exist, without writing metadata.
  A real copied witch through `AssemblyLine.make_avatar3d(... reuse_mesh=True)` took 27.8s,
  kept 99,982 triangles, six hair chains and 16 painted expression entries, and matched source
  mesh/original avatar SHA256. Copy lives under ignored `outputs/3d-quality-refresh/witch-copy`.
- **Safe replacement:** `_replace_avatar` moves the old folder to a unique backup before a swap,
  restores it if the new folder rename fails, and leaves the named backup recoverable with a
  clear error if restoration also fails. The 3D path checks cancellation after CPU rigging and
  before the swap; cancellation/failure never intentionally deletes the previous avatar.
  Gallery now opens `/generator?model=<id>` after 3D queue submission so a very fast CPU refresh
  that finishes before navigation still shows its result (previously Studio only reattached to
  running jobs and could appear blank).
- **Stage expression controls:** available painted A/I/U/E/O expressions can be selected for
  mouth motion; no automatic phoneme recognition claimed. Eye emotes are mutually exclusive and
  fade during blinks; > < suppresses regular blink decals. Mouth emotes are mutually exclusive
  and suppress speaking mouth shapes, while independent cheek emotes combine. Paired eye artwork
  remains one decal and fades together for winks. Failed VRM/file replacements restore the prior
  selection, model and correct download URL. Existing VRM0/GLB/Vivi behavior remains supported.
- **Engine hardening:** port preflight rejects any occupied listener, the engine must print its own
  post-bind startup message before requests (latched even if later logs roll out of the tail),
  forbidden engine args cannot override private paths,
  30s history and 60s prompt calls remain cancellable, engine output GLB paths are constrained to
  its private output folder, staged inputs are cleaned after failures, and child-exit races do not
  hide the actual error. `tests/test_mesh3d_safety.py` covers these with a fake engine; no real
  GPU generation was rerun per the owner. Keep this limitation visible until a real 3D job succeeds.
- **Verification:** 146 GPU-free pytest, Node preview/tagcomplete/3D-tracking suites, and 7 Edge
  journeys (`test_3d_stage.py`, `test_3d_refresh.py`, `test_3d_expression_quality.py`) passed.
  Browser tests cover Gallery refresh, source/seed preservation, VRM/GLB import, external-resource
  rejection, mobile layout, mouth shapes, blink-aware/exclusive expressions and failed replacement.
  Screenshot `outputs/3d-expressions-review/witch-expressions.png` is a browser QA image.
  Final real server is on 127.0.0.1:5055, launcher PID 19676 (queue empty before restart), logs
  `outputs/vrm-review/quality-server.*.log`. Live Gallery refresh button and witch mouth choices
  checked without invoking a job; `outputs/3d-expressions-review/witch-stage.png` is the live stage. A physical
  webcam and third-party VSeeFace/Unity import remain untested. Changes and owner work uncommitted.

3D motion and input fidelity (2026-10-28, uncommitted):

- Owner authorized GPU tests and requested better-looking, more responsive 3D avatars.
  `mesh3d.frame` resizes the cut-out in premultiplied RGBA before compositing against black,
  rather than resampling the original background into the edge. Crop geometry and the mask
  are unchanged. Tests cover the halo; comparisons using saved character masks found changes
  at 8,430–13,541 edge-near pixels on three characters. This is an input correction, not a
  demonstrated guarantee that the finished mesh is visually better.
- The 3D stage now uses frame-rate-independent, bounded head and smaller delayed torso follow,
  varied quiet idle motion, a blink with a closed hold, and separate mouth attack/release.
  Idle resumes after tracking drops out. This is face tracking and inferred torso motion, not
  full-body tracking. `static/js/3d_motion.js` contains the math; a new Node suite and browser
  tests cover head/torso motion, independent eyelids, mouth and synthetic tracking dropout.
- `rig3d.py` retains a painted closed neutral mouth when the original art has an open mouth
  and painted vowels; mouth expressions move that neutral decal behind the face. The exact
  open-mouth-plus-vowels scenario was tested synthetically, not on an existing character.
  T-pose rotation handles opposite arm directions and fails clearly for a missing direction.
- Two sequential real Pixal3D runs used the witch artwork and identical seed 1052685790 in
  separate ignored folders. The pre-fix run completed in 324.7 s and verified the hardened
  engine startup/polling; the revised framing run completed in 453.6 s with silhouette fit
  0.983, 99,982 triangles and 6 hair chains. The observed maximum VRAM sample was 6,575 MiB,
  not a guaranteed peak; memory briefly dropped to about 1 GiB available during loading. GPU
  returned to 0 MiB. The revised mesh was CPU-rigged; output, logs and stage screenshot are in
  `outputs/3d-gpu-review-edgefix-20261028/`; the originals were never replaced. The screenshot
  shows remaining facial/blink and cropped-hat artifacts; inspect and improve those, do not
  claim professional output or that the edge correction alone improved the final rendering.
- Verification: 149 GPU-free pytest tests, four Node suites, and 9 Chrome 3D browser journeys
  passed, including the actual newly generated VRM and synthetic tracking. VRM 0.0 official
  schema validation: 0 errors on that VRM. Current Edge exited on launch through Playwright;
  the browser tests use Chrome and the newer headless mode. No physical webcam or external VRM
  app was tested. Real server restarted with an empty queue at 127.0.0.1:5055, launcher PID
  41868, logs in the review folder. No commit or push.

Preview ribbon and hair cutouts (2026-11-01, uncommitted):

- Owner reported the ribbon on silver-haired `82c64229-de47-435c-bce7-759724a5e096` leaves
  a ribbon-shaped hole as it sways. The original See-through back and front hair both have
  transparent holes at the ribbon's rest position; the headwear layer moves independently.
- Preview PSD v17 repairs small gaps under later, nearly opaque (`alpha >= 250`) headwear in
  front/back/side hair. A bounded morphological close estimates the hair silhouette; local
  normalized colour interpolation fills only existing transparent pixels under the accessory.
  Translucent edges, outside-hair regions, visible pixels, original assets and artist exports
  are untouched. PSD cache rebuilds lazily; no GPU generation is needed.
- On the owner's model the repair filled 7,919 back-hair and 6,497 front-hair pixels. The
  before/after preview composite at rest differed by at most 5 channel levels because most
  See-through accessory interiors have alpha 254. Chrome rendered the real preview with 26
  layers and headwear moving under head turns, without page errors. Screenshot and layer
  comparisons: `outputs/ribbon-review/`. Minor fine-edge artifacts remain possible at
  extreme poses; fills interpolate hidden hair, they do not reconstruct individual strands.
- Verification: 150 GPU-free pytest tests pass, including a new regression for concealed
  holes, translucent accessory edges, unrelated gaps, silhouette and source immutability.
  No GPU work or saved studio-setting changes. Real server started at 127.0.0.1:5055 after
  confirming no queued jobs; launcher PID 37068, logs in `outputs/ribbon-review/`.

Lively pages, gestures and quality of life (2026-11-01, Claude, uncommitted):

- Owner: "make the pages more lively and unique and continue to develop model outputs and
  animations", then "keep working adding Quality of Life changes until all session credits are used".
  No GPU run was needed: browser-side work, small API additions (favorite, reveal, thumbnails,
  rig_outdated), a cross-site POST guard, and CPU-side rig3d output fixes.
- **3D gestures** (`3d_motion.js` `gesture(name, t)`, applied in `3d_studio.js`): Wave, Nod, Shake head,
  Hop, Bow, Cheer buttons + keys 1–6, layered over follow-through and camera tracking. Arms are given
  as elevation-from-horizontal + elbow bend; `rigInfo()` measures in world space which way each upper
  arm rises and which way the spine bends forward (VRM0 and VRM1 flip x/z rotations), so the same
  numbers work on Ka-myii VRM0 avatars, Vivi and already-lowered artist poses. Hop moves the hips by a
  share of the avatar height. **Little gestures** (default on): while idle with no tracked face, a
  nod/wave/hop every 16–34 s. Gotcha: rAF timestamps can be earlier than the click's
  `performance.now()`; the first frame then saw negative elapsed time and cancelled the gesture
  (now clamped). Checked on idol 72213aa1: wave/cheer/bow look right (bow tips forward);
  the hip sliver under a raised arm is the known fused-hand mesh artifact, not the gesture.
- **3D stage**: background picker (Studio, Green/Blue screen for OBS chroma key, Dark, Light,
  Checkerboard; PNG export stays transparent), Turntable (OrbitControls autoRotate), Fullscreen,
  reopens the last viewed Ka-myii avatar when there is no `?model=` (localStorage `kamyii.3d`, which
  also keeps the background, framing, Idle and Little gestures).
- **2D Live Preview gestures** (`lib/motion.js` `gesture`, `lib/app.js`): Nod (N), Shake head (H),
  Huh? tilt (T), Yay (Y), Shy (L); additive pose targets on simulation time (pauses with the preview),
  listed in the Shortcuts dialog. Button gestures are not relayed to OBS (an OBS window plays its
  own Little gestures). `app.js?v=m4`; the vendored UPSTREAM.md notes the change.
- **Gallery**: filter chips with counts (All / Ready / Sketches / 3D avatars / Needs attention; empty
  groups hide), sort (newest, oldest, name), remembered per browser (`kamyii.gallery`), `/` focuses
  search, Esc clears it; 3D and Blink & talk badges on the art; hover lift + breathing art; More options
  → **Make a variation** (`/generator?brief=<id>`: same brief/settings/toggles, seed -1, no result shown).
  The list reloads by itself when the nav sees a job finish (`kamyii:job-done` event).
- **Nav (every page but the Studio)**: polls `/api/generation/jobs` every 4 s; shows "Creating · 42%"
  (+ queued count) linking to the Studio, then "Character ready ↗" linking to the new character. The
  tab title follows ("42% · Ka-myii — Collection", "✓ Ready · …" if it finished in a background tab).
- **Studio**: ✦ Surprise me (random VTuber-friendly tag brief), unsent prompt/negative draft kept in
  localStorage (`kamyii.studio.draft`, restored only without `?model=`/active job), Create scrolls the
  progress into view. `fillForm(saved)` is now shared by reopen and variation (also restores toggles
  and 2D + 3D).
- **Home**: the character sheet cycles through up to six separated characters (crossfade every 5 s,
  pauses on hover/hidden tab, off with reduced motion), shows the name and No. x / y; a Recent cast
  strip of the eight newest characters links to Live Preview or the Studio.

- **3D output** (`rig3d.py`): VRM emotion presets **Joy** (closed eyes + blush), **Angry** (anger vein),
  **Sorrow** (tears), **Fun** (star or heart eyes + blush), each only when the painted parts exist —
  VSeeFace-style apps bind hotkeys to these; the stage labels three-vrm's happy/sad/relaxed as
  Joy/Sorrow/Fun and groups Joy/Fun with the eye emotes. Closed-eye decal fixes: fact 33. Checked on a
  CPU re-rig of a **copy** of the witch (`outputs/qol-review/witch-emotions/avatar.vrm`, 27 s, 99,982
  triangles, 6 chains, 18 shapes): VRM 0.0 schemas 0 errors, 3D e2e on it pass; before/after shots
  `outputs/qol-review/joy-before.jpg` / `joy-after.jpg`. **The owner's avatars were not changed**:
  Gallery → More options → Refresh rig & expressions applies this to each (CPU, ~30 s).
- **Stage framing**: Full body / Bust / Face presets sized by shoulder width (Ka-myii's head pivot
  sits near the chin, so the aim is above it), remembered per browser; idle **breathing** (chest lift
  + shoulder rise, `motion.breath`).
- **Studio**: drop an image anywhere on the page or paste one (Ctrl+V) to import it.
- **Verification** (latest): 162 GPU-free pytest and the 4 Node suites pass. Every browser e2e file
  passes with the final code — Studio/Gallery journeys + `test_qol.py` (34; the import/edit journey
  needed the "Paint scrolls" fix below) and the 3D files (10 + 2 skips: the optional licensed
  Vivi sample, and the clip-size check in headless Chrome). Run 3D e2e with
  `KAMYII_E2E_CHANNEL=chrome` and `KAMYII_TEST_GENERATED_VRM` = the witch (the expression test needs
  her toggles; the idol fails it by design). One earlier full run was stopped by Claude Code for low
  system memory (not a failure); later runs were split per file. No stray output folders after any
  run. Real server on 127.0.0.1:5055 (logs `outputs/qol-review/`); the owner's :5000 server was not
  touched. No commit or push.
- **Rig version**: `rig3d.RIG_VERSION` (now 4, drawn face marks; 3 was chest springs) is written to `rig.json`; `/list` adds
  `avatar3d.rig_outdated` when a refreshable avatar was rigged by an older version, and the Gallery
  card then shows **Update 3D rig ✦** (the same CPU refresh, ~30 s). Bump it whenever a refresh
  improves existing avatars. All three of the owner's avatars currently show it.
- More small things: Live Preview **Little gestures** toggle (`auto.lively`, saved in settings);
  rotating tips under the Studio progress bar; Studio tab title says "✓ Ready" / "✕ Failed" when a
  job ends in a background tab; Gallery dates are relative ("4 days ago", exact time on hover) and the
  art's tooltip is the prompt; the 3D stage header is shorter so the stage fits an 850 px screen;
  the Studio remembers whether Advanced is open; Home's three steps link to the Studio, Gallery and the
  latest character's Live Preview; dropping an exported .zip on the Gallery reopens it.
- **3D microphone lip sync** (no camera needed): **Use microphone** on the stage reuses
  `RigDevices.createMic` (level → mouth) and `RigVisemes.classify` (vowel weights, blended at 18/s),
  spread over the avatar's painted A/I/U/E/O mouths when it has more than one, otherwise the chosen
  mouth shape. Tested with an oscillator stream standing in for `getUserMedia`; no real microphone.
- **Studio rename**: the result shows the character's name as an editable heading (Enter saves via
  `/rename`, Esc reverts). Fixed `.rename-input` / `.result-name` losing to the generic
  `input[type=text]` rule (the Gallery rename box had been 12 px instead of 22 px).
- Home hero has a few slow ✦ sparkles (CSS only; off with reduced motion). README documents the
  gestures, stage tools, emotion presets and Studio/Gallery conveniences.
- **Thumbnails**: `/preview/<id>?size=thumb` serves a 640 px WebP cached as `<character>/thumb.webp`
  (remade when the preview/base image is newer; served from memory because Windows cannot replace a
  file another response still holds open). Gallery and Home use it: 13.1 MB → 0.63 MB for the owner's
  15 characters. The Studio canvas still loads the full PNG. Thumbs travel in export zips (~40 KB).
- 3D page with characters but no avatars shows "Make one from a Gallery character ↗". e2e covered.
- 3D stage keys: Shift+1–9 toggle expression chips (like Live Preview), M toggles the microphone. The
  first nine chips show their key (`⇧1`… in an aria-hidden `<kbd>`, so accessible names are unchanged).
- Create with an empty prompt now flashes the box and says what to do (Surprise me / Recent) instead
  of only moving focus. e2e covered.
- Studio "Expression toggles" label counts the selection ("2 selected, about 30 s"; ~15 s each).
- Studio **Clear** (next to Recent / Surprise me): forgets the remembered brief and reloads with the
  default settings (asks first when a prompt is typed). e2e covered.
- 3D **Record clip**: MediaRecorder on the canvas (VP9 WebM, 8 Mbit/s, max 15 s, `kamyii-3d.webm`);
  the chosen background is set as `scene.background` while recording (green screen pixels measured
  exactly #00b140), then the canvas goes back to transparent. 2 s ≈ 0.8 MB. Hop on busts (< 1 m tall)
  lifts 35% as far (the witch bobs 8 mm instead of floating). Gotchas: setting `scene.background` in
  the same task as `captureStream()` made the **first** recording on a page empty (no frames) — the
  background now switches on the next animation frame; the blob URL lives 60 s so slow downloads
  finish. Headless Chrome + SwiftShader captures **no** frames from an on-page WebGL canvas (a
  detached one works), so the e2e test checks the flow and skips only the size check there.
- **Export** leaves out `live_preview/` (the preview PSD cache — its key includes file times, which
  unzipping changes, so it could never be reused; 5–12 MB per character), `thumb.webp`, and
  `avatar3d.new` / `avatar3d.backup-*` work folders (`_download_zip(..., skip=...)`, tested).
- Studio result **Open folder** too (e2e test intercepts the request — never let a test hit `/reveal`
  for real, it opens Explorer on the owner's desktop).
- Gallery More options → **Open folder** (`POST /reveal/<id>`, `os.startfile` / `open` / `xdg-open`),
  only for a browser on the same machine; covered by a test with the opener mocked (never clicked
  for real here, it would pop up Explorer on the owner's desktop).
- **Security** (fact 20 addendum): cross-site POSTs are refused.
- **Favorites**: ☆/★ on each Gallery card (`POST /favorite/<id>` {favorite: bool} → `metadata.json`
  `favorite`, so it travels with exports) and a ★ Favorites filter chip (hidden until something is
  starred). The star sits on the card, not inside the art link.
- **Quick look** (Gallery; swipe left/right on touch screens too, checked at 390 px): ⤢ on each card opens a native `<dialog>` with the full-size art;
  ←/→ flip through the current filtered/sorted list (wrapping), Esc or the backdrop closes it, with
  Artwork & layers / Live Preview links, plus ★ Favorite and Separate layers (sketches/failed; queues
  `/retry/<id>` and stays in the viewer) for triaging a batch. The grid re-renders on close. The art
  link itself still opens the Studio. e2e covered.
- Studio result **Make 3D avatar** (when the engine is installed or in demo mode, and the character
  has none yet): queues `/avatar3d/<id>` and follows the job; the failure note now points at it.
  e2e covered. Also **Add blink & talk** (See-through characters without painted expressions, same
  rule as the Gallery): visibility checked on real characters (b6f6fe2f shows both, the idol neither);
  not clicked (a real GPU job) and not e2e-testable in demo mode (no See-through backend there).
- **Character notes**: `POST /notes/<id>` {notes} (printable + newlines, 500 chars; empty clears) →
  `metadata.json` `notes`. Gallery More options → Notes edits in place (Ctrl+Enter/blur saves, Esc
  cancels); the note shows under the name (two lines) and search includes it. Rename in More options
  is no longer red (that colour is for Delete). Unit + e2e covered. Quick look shows the note under the
  name; Home's recent-cast cards use the note (or prompt) as their tooltip.
- **Status panel**: clicking the nav pill (when no job link) opens `GET /system` — mode, See-through and
  3D engine installed, character count/size, trash count/size, free disk, outputs folder — with
  **Empty trash** (`POST /trash/empty`, browser confirm first). Owner's machine on 2026-11-01:
  15 characters = 627 MB, **14.5 GB free disk**; agent review folders in `outputs/` total ~450 MB
  (not counted; left alone because this file references some of them).
- Live Preview header: **Gallery** link, and **3D avatar ↗** when the character has one
  (`has_avatar3d` from the `/preview/<id>` route).
- **Separate all starred sketches**: in the ★ Favorites view a button queues `/retry/<id>` for every
  starred sketch (sequential POSTs, "Queued n of m"). Batch workflow: How many 8 → quick look and star
  the keepers → Favorites → Separate all. e2e covered (two demo separations take ~100 s). Queued ids
  are remembered for the page (`separating`), so neither the bulk button nor quick look's Separate
  offers the same sketch twice (a separation creates a new character; the sketch stays a sketch).
- **Whole-brief draft**: `kamyii.studio.draft` now stores the brief shaped like a saved request
  (prompt, negative, style, framing, checkpoint, sampler, steps, CFG, size, clip skip, LoRA, toggles,
  2D + 3D — never the seed) on every form change/chip click, and `restoreDraft` hands it to
  `fillForm`; older `{prompt, negative}` drafts still load. All 9 journeys pass with it.
- **Delete is recoverable**: `DELETE /delete/<id>` moves the folder to `outputs/_trash/<id>` (with a
  `.deleted` timestamp; trash older than `TRASH_DAYS` = 7 is purged on later deletes), `POST
  /restore/<id>` moves it back; the Gallery dropped its confirm dialog for an 8 s **Undo** toast. Ids
  starting with `_` are refused (the old endpoint would have erased `outputs/_uploads` if asked).
  `/list` skips `_trash` (no metadata.json at its top). Unit + e2e covered. Disk space is freed after
  a week, not immediately — tell the owner if space matters.
- Gallery scroll position survives Back (saved in sessionStorage on `pagehide`, restored once after the
  first render on a `back_forward` navigation). The e2e test passes, but Chrome's back/forward cache
  may be what restores it there; the code covers reloads without that cache.
- Gallery More options → **Copy prompt** (clipboard). Studio errors: CUDA out-of-memory messages
  (`out of memory`, `CUDA_ERROR_OUT_OF_MEMORY`, `CUBLAS_STATUS_ALLOC_FAILED`) are shown as advice
  (close GPU apps, try again, Retry layers if the artwork exists). e2e covered with a mocked job.
- **Paint scrolls the picture into view**: the result name heading pushed the edit panel down, so
  opening "Edit a region" left the canvas above the viewport (the import/edit journey's mouse strokes
  landed at y < 0 and nothing was painted). Turning Paint on now scrolls the mask canvas into view.
  (The 404s for `/layers/<id>` on sketches in that journey are pre-existing and harmless.)
- **Jobs in progress** (Gallery): a strip above the grid with each running/queued job (oldest first):
  prompt, state/progress bar, and the artwork thumbnail once `artwork` is known; polls `/jobs` every
  3 s while something is active, 10 s otherwise. Note for tests: the Gallery now polls `/jobs` too, so
  a mock must not count calls (the nav test was rewritten to flip the job's state explicitly).
- **? shortcuts list**: base.html has a `<dialog id="kmHelp">` filled by each template's
  `{% block shortcuts %}` (Studio, Gallery, 3D; Live Preview keeps its own), opened with `?` or the
  footer link. The Studio's Esc-to-reset now ignores Esc while a dialog is open or while typing in
  any field (it used to hide the result when closing things or editing the prompt). e2e covered.
- **Artwork while separating**: once the base image is saved, the pipeline records
  `ProgressManager.artwork[task_id] = model.id` (dropped by `forget`); `JobQueue.describe` adds
  `artwork` to a running job, and the Studio's progress overlay shows its thumbnail during the ~10 min
  See-through stage, so a miss can be cancelled early. Unit + e2e covered (e2e mocks the job API).
- **How many** (Studio, prompt mode): 1/2/4/8 queues the rest of a batch behind the first job, one
  seed each (a set seed counts up: 100, 101, …; random stays random). A failure posting an extra job
  shows in the queue note without ending the first job's progress view. e2e covered. Its hint shows
  rough totals (sketch ~1 min, layers ~11, 2D + 3D ~15 per character). The prompt box grows with
  long briefs (`field-sizing: content`, up to 340 px; browsers without it keep the fixed box).
- Studio queue panel **Cancel all queued** (shown with two or more waiting jobs; the running job and
  the one on the canvas are untouched). e2e covered with a mocked job API.
- **Recent briefs**: Studio ↺ Recent lists the last eight distinct prompts from the collection
  (newest first, uploads skipped); picking one fills prompt + negative and the draft.
- **Test isolation leak found and fixed**: the front-redraw journey ends by clicking "Try another seed";
  that job kept running after the test, the autouse `OUTPUT_DIR` patch was undone, and it wrote a
  half-made character (metadata.json only, the bard test brief) into the **real** `outputs/`
  (`13eb1bbf…`, removed). The `base_url` (journeys/QoL) and `stage` (3D) fixtures now wait up to 60 s
  for queued/running jobs before shutting the server down. The same thing happened in unit tests
  (a new security test of mine queued a real demo job: `7d79f9f6…`, prompt "1girl", removed), so the
  autouse `_private_outputs` fixture in `tests/conftest.py` now also waits for the shared
  `generation.job_queue` to go idle before the patch is undone. If a stray folder with only a test
  brief appears in the Gallery, this is where it came from; check `outputs/` after test runs.

### Chest physics (2026-11-02, Claude, uncommitted)

Owner asked for "boob jiggle physics" (standard on VTuber models). Three places:
- **3D/VRM**: chest spring bones (fact 34), so VSeeFace and other VRM apps bounce the chest too.
  `RIG_VERSION` 3, `rig.json` `bust_springs` (0 or 2). Re-rigged copies: `outputs/qol-review/idol-bust/`
  (2 springs, 0 VRM schema errors) and `witch-emotions/` (0: cropped bust). Weight pictures
  `outputs/qol-review/*-bust-weights.png`. The owner's own avatars get it via **Update 3D rig ✦**.
- **3D stage**: **Chest bounce** slider (shown only when the avatar has `bustL/R` springs; 0 = still,
  50% = as rigged/what VRM apps play, 100% = stiffness × 0.45 and drag × 0.4), remembered in
  `kamyii.3d.bounce`; `Kamyii3D.state.chestSprings` reports stiffness/drag/swing. On the idol a Hop
  swings 0.14 rad at 50% and 0.31 rad at 100%, settling in ~1 s; no tearing seen at the maximum.
  Capturing that headless needs a virtual clock (SwiftShader frames take seconds; the gesture is
  over before a screenshot) — scratchpad `bounce_shots.py` overrides `performance.now`/rAF.
- **Live Preview (2D)**: the old jiggle was hidden under "Breathing sway"/"Breathing origin" (same
  `bust`/`bustY` settings, now labelled **Chest bounce**/**Chest height** with a hint). Default
  `bust` 0.4 → 1 (the witch moved ±3 px on Yay before, ~±7 px now); nods (`nodPose.body`) also drive
  it, and a second spring (`bounceX`) sways it sideways with the torso arc. `Anime25D.state().pose`
  has `chestY`/`chestX`. Characters with a saved `bust` keep their value.
- **Cubism scaffold**: `auto_physics` adds a "Bust" pendulum → `ParamBustX` (Cubism standard id)
  when there is a `topwear` layer; it does nothing until the artist rigs that parameter.
- Not verified: VSeeFace/Unity playback, a flat-chested or male character (all three avatars are
  1girl; the springs are added regardless of shape — the 3D slider or 2D 0 turns them off), real
  head tracking driving the bounce (only Hop/Yay/nods were measured).
- Verification: 164 GPU-free pytest (two new rig tests + the Cubism test extended), 4 Node suites,
  3D e2e 8 + 2 skips (new: a stick figure rigged with chest springs bounces on Hop and the slider
  stiffens it), journeys 9/9, and the Live Preview gesture test now also checks the chest bounce.

### Chest-motion refinement (2026-11-03, Codex, uncommitted)

- Owner requested a visual/performance review of Claude's 2D and 3D chest physics. In the 2D
  preview, chest influence is now evaluated in each topwear layer's original artwork coordinates
  and cached until chest height or anchor changes. Previously the Gaussian was recomputed per
  vertex on every frame using already-deformed positions, so it could drift across clothing.
  A smooth per-axis limit of 12% of face height applies after the strength slider, keeping
  abrupt extreme inputs bounded while preserving small motions. Zero strength skips this work;
  anchor editing forces a still chest. Existing settings and source art are unchanged.
- 3D's zero slider formerly set stiffness/drag but still lagged on a hop. At zero the stage now
  resets the two named chest joints after each simulation update; hair joints remain active.
  Re-enabling the slider resumes their motion. This changes stage playback only, not VRM rig
  generation, export, or spring values used by other applications.
- Verified 164 GPU-free pytest, Node preview and 3D-motion suites, and focused Chrome 2D and 3D
  browser tests. The 2D test exercises rendered topwear with the control at 0/1/4; the 3D test
  checks a hop, zero during a hop, and re-enabling. Existing silver `82c64229` and witch
  `004b7e89` previews rendered without page errors; sampled raw 2D spring ranges on Yay were
  -5.69…7.04 and -5.11…6.15 px respectively, before the strength/limit. The separately saved
  idol VRM `outputs/qol-review/idol-bust/avatar.vrm` showed spring movement on Hop and zero
  when disabled, with no page errors. Images/scripts in `outputs/chest-review-codex/`.
  These are browser/gesture checks, not physical webcam testing or a measured FPS benchmark.
  The VRM's behaviour in external apps and subtle artifacts across other body shapes remain
  unverified. The local server was restarted after confirming an empty queue, launcher PID
  19964, logs in that review folder. No original character or saved setting was changed.

### Moods, face marks, animal ears & tails (2026-11-03, Claude)

Owner asked for "blush, gloom, smug, angry, and other toggles … as well as better tails and animal
ears", then to push to main when satisfied.
- **Live Preview moods** (`app.js` presets): Smug (key 8), Angry (9), Sad, Gloomy, Flustered, Pouty.
  Brow angle sign: `browAngSym` > 0 drops the inner ends (angry), < 0 lifts them (worried); eyebrow
  side L is the *image* left (rigger splits by centroid vs face centre). Brow angles and marks are
  `PRESET_EXTRAS`: not reset by every preset; a preset remembers the ones *it* sets (`presetBefore`)
  and puts them back when cleared or replaced (after a reload, back to defaults). A frowning closed
  mouth bends ×8 instead of ×3 (drawn mouths are mostly smiles; ×3 at −0.85 moved the corners ~2 px).
- **Face marks**: `mark blush L/R`, `mark gloom`, `mark anger`, `mark sweat` are companion-style
  layers from `Motion.markArt` (canvas-drawn, MIT), placed each frame by `markBox()` from the eye and
  face anchors. Anger and sweat go by the **eyes**, not `A.face.y0`: the face layer runs up under the
  bangs/hat (on the witch the anger mark first landed on her hat). Params `markBlush/Gloom/Anger/Sweat`
  (0/1, fade with the usual smoothing), buttons `[data-mark]`, keys R/G/A/D, saved like other params,
  `Anime25D.state().marks`. Painted toggles are unchanged and separate.
- **Ears/tails 2D**: `neko` 0–5 = off/cat/fox/wolf/bunny/bear, `tail` 0–4 = off/cat/fox/wolf/devil
  (`Motion.EARS/TAILS`; old saves' 1 = cat). Ear canvases are now 128×320 (root y 311, the cat drawn in
  the bottom half, so old fits look the same); `image.top` tells `updateStageBounds` where the drawing
  starts so short ears don't pad the stage. Fox/wolf tails draw wider (×1.7/×1.45); idle sway breathes
  with an occasional flick; tail phase speed per kind. Art is redrawn when design or fur changes
  (`motion.palette` now keys palette+ear+tail; art runs before the bounds).
- **3D**: drawn marks in every VRM (fact 35, rig 4 → Gallery offers **Update 3D rig ✦**); three-vrm
  maps joy/angry/sorrow to `happy/angry/sad` chips. New `static/js/3d_companions.js`: stage-only ears
  (extruded outlines on the raw head bone, placed by raycasting the skinned mesh from above) and a
  rebuilt-every-frame tube tail on the hips (needs a full body; a few heights are tried because the
  test stick figure has a gap at the hip joint). **Texture images are not readable after upload**
  (`map.image` empty in r180), so "Match hair" reads the rendered stage pixels (`stagePixel`,
  `preserveDrawingBuffer`) at probes beside the face sized by shoulder width (the crown is a hat on
  the witch; two probes once hit a white hair highlight, so it's a median of up to 12). Settings per
  character in `kamyii.3d` → `companions[source id or file name]`. `Kamyii3D.state.companions`.
- Owner's characters/avatars untouched. Re-rigged review copy: `outputs/qol-review/idol-bust/`
  (rig 4: blush/gloom/anger/sweat + 2 chest springs). The old VRM schema validator in the scratchpad
  was gone; the check was three-vrm loading it with no page errors (structure not re-validated
  against the JSON schema this time).
- Verified: 165 GPU-free pytest (new: drawn marks / painted blush kept), 4 Node suites (new: ear/tail
  menus match `EARS/TAILS`), journeys 9/9, QoL 28/28 (new: Angry adds and removes its mark and brow
  angle, a self-chosen blush survives Gloomy, bunny ears + devil tail render), 3D 11 + 2 skips + refresh
  (new: fox ears/tail fit, sway, pink "match hair", remembered after reload). Visual sheets of the witch
  (2D moods/marks/ears) and idol/witch (3D marks, ears, orbited tails) were inspected.
- Not verified: VSeeFace playback of the new expressions; 3D ears on unusual heads (hats put them on
  the hat, as in 2D — Ear height slider); ears/tail on a webcam-tracked stream; the smug brow tilt is
  hidden under bangs on both test characters.

## 11. How to work in this repo

- Smallest complete change; reuse existing code and libraries; no speculative abstractions.
- Every non-trivial logic change gets a small GPU-free pytest (monkeypatch GPU parts like
  `tests/test_pipeline.py` does). Demo mode must keep working without torch/GPU.
- For anything touching the GPU path, verify on the real GPU once, watching `nvidia-smi` for
  spills. Never run two GPU jobs at once; kill stray `python.exe` processes you started.
- Verify UI changes in a real browser, not just by reading code.
- Report honestly: what you verified, what you didn't, and measured numbers.
