# 3D avatar roadmap

Ka-myii's next goal is a locally usable 3D VTuber character: an anime-style, textured,
rigged model with expressive face shapes, motion and an interoperable export. This is a
separate path from the existing layered 2D preview. A reconstructed mesh is an important
step, but on its own it is not an animated or tracking-ready avatar.

## Hardware and product constraints

The target is Windows 11 on an RTX 4070 Laptop GPU with 8 GB VRAM and 16 GB RAM, often with
other applications open. Run generation, rigging and tracking in separate stages; unload
one model before the next, and measure both GPU memory and available system RAM. NVIDIA's
shared-memory fallback can make an apparent success unusably slow. Preserve existing Ka-myii
projects and exports. Optional components should be installed by Ka-myii and have their
own pinned dependencies; do not require ComfyUI or another preinstalled application.
Weights belong in the cache or ignored engine folders, never Git. Track source versions,
model and asset licenses, downloads, and precise limitations in README and AGENTS.md.

The following sequence started as a proposed plan. **Status (2026-10-28):** a first version of
stages 2–4 works end to end on this laptop — **Make 3D avatar** (Gallery) and **2D + 3D** (Studio)
turn a character into a rigged VRM 0.x avatar in about 4 minutes, peak 7.0 GB of VRAM; see
"Implemented path" below. It has been checked on one real character in Ka-myii's own three-vrm
stage, not yet in VSeeFace, Unity or VRoid Hub.

## Implemented path (2026-10-28)

| Stage | What Ka-myii does | Measured |
|---|---|---|
| Mesh | Cut-out framed by Ka-myii, Pixal3D int8 (Comfy-Org repack) in a private pinned ComfyUI 0.42.1 (`engines/comfy3d`, from Atelier3D's design), remesh/decimate to ~100k triangles, 2K bake, the artwork projected back onto the front | ~3 min + ~30 s engine start; 6.99 GB peak VRAM |
| Joints | DWPose whole-body keypoints on the artwork, lifted into the mesh along Pixal3D's camera rays | < 1 s |
| Skinning | Torso by height; arms/legs from See-through layer labels spread over the surface (capsules without layers); fused hand contacts cut | ~5 s |
| T-pose | A-pose arms raised with the weights (VRM apps expect a T-pose rest) | < 1 s |
| Face | Painted closed eyes / open mouth as morph-target decals: Blink, Blink_L/R, A (I/U/E/O partway or painted visemes) | < 1 s |
| Hair | Spring-bone chains for hair below the jaw, head/neck/chest colliders | < 1 s |
| Export | VRM 0.x (humanoid, MToon, blend shapes, secondary animation, thumbnail, restrictive default license) | 7.5 MB |

Why not the candidates below: TripoSR needs a native CUDA build on Windows and produces a softer,
vertex-coloured mesh; Pixal3D was already installed and validated by Atelier3D, needs no compiled
extensions and is pixel-aligned with its input — which is what lets the rig and the face decals
use the artwork directly. UniRig / Make-It-Animatable remain options for non-humanoid or
unusual bodies; the heuristic rig is specific to front-facing humanoid characters.

## Quality pass (2026-10-28)

The Gallery can now refresh a rig and its painted expressions from an existing 3D mesh on the
CPU. This retains the mesh and its seed and is useful after improving the rig or painting new
expressions. Hats, ears and earrings remain rigid to the head, and projections outside the
artwork no longer inherit edge labels. The 3D stage supports available A/I/U/E/O mouths and
avoids stacking incompatible eye/mouth emotes. The current paired eye overlay still fades on
both sides for a wink; independent decals remain a useful next step.

The private engine's lifecycle and path handling were hardened with simulated tests. A real
GPU run of those changes remains outstanding at the owner's request. Rigging was rechecked on
copies of three existing meshes; these checks do not establish quality for all outfits, fused
limbs or generated backs. Physical webcam and independent VRM-app import still need review.

## Original plan

## Stage 1 — VRM import, rendering and face tracking

A first local 3D stage is available at `/3d`; it has been exercised with a VRM 0 sample.
Physical webcam tracking and VRM 1.0 still need independent validation.

Provide a locally hosted 3D stage that opens a user's existing `.vrm` file, renders its
humanoid mesh and animation features, and supports neutral camera framing, orbit, zoom and
optional idle motion. A separately controlled webcam may drive supported head movement,
blinks and mouth expressions. Only claim an expression when the imported model supplies it;
models have different bones and expression sets. Keep camera and microphone off until the
user starts them. Inform the user if the model has no humanoid rig or is missing a requested
expression. Spring movement should use the model's actual spring-bone data when available.

[three-vrm](https://github.com/pixiv/three-vrm) supports VRM on Three.js and is MIT-licensed;
vendor a pinned, locally served build and its licenses. Existing Ka-myii MediaPipe FaceMesh
can supply basic face signals; Google's newer [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/web_js)
can supply additional blendshape scores in a later iteration. A single face webcam does not
provide full-body or finger tracking. Test with a real VRM and browser rendering, camera
permission/cancellation and locally served assets; a synthetic landmark test is not a physical
webcam validation. VRM 0.x and VRM 1.0 compatibility should be tested separately.

**Test reference:** pixiv identifies beta AvatarSample_2, Vivi, as
[CC0](https://vroid.pixiv.help/hc/en-us/articles/360014900273). A copy is available at
[`vroid/beta/Vivi.vrm`](https://github.com/madjin/vrm-samples/blob/master/vroid/beta/Vivi.vrm).
Its embedded VRM 0 metadata reports `title: Vivi`, `licenseName: CC0`, and
`allowedUserName: Everyone`. The file observed during research was approximately 18.1 MB.
For repeatable tests, preserve source URL, checksum and the official license reference; do
not assume every VRoid sample has these terms. VRM 1.0 testing needs a separately vetted
fixture. Each user's own VRM retains its embedded author and use conditions; see the
[VRM metadata specification](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/meta.md).

## Stage 2 — artwork to an inspectable 3D mesh

Start from a saved Ka-myii character or imported artwork, preferably a complete full-body
A- or T-pose on a plain background. Generate a textured mesh as an optional separate job,
then let the owner inspect front, sides and back and download the mesh. A single image leaves
occluded surfaces ambiguous; cloak backs, hair, faces, hands and accessories may be invented
or merged. Treat this as a draft for review, not a faithful replica or animation-ready model.
Consider optional front/side/back references later.

[TripoSR](https://github.com/VAST-AI-Research/TripoSR) is a candidate for a first experiment:
its source and [weights](https://huggingface.co/stabilityai/TripoSR) are MIT-licensed and its
upstream documentation reports about 6 GB VRAM for default single-image inference. This is
an upstream number, not a measurement on the owner's laptop; texture baking and other
processes can add memory. Its native dependencies and Windows installation need a clean,
repeatable test. Successful generation must produce a loadable mesh and preserve source art;
a generated GLB alone does not contain a usable humanoid skeleton, facial shapes or VRM
metadata. Run it in an isolated, cancellable engine process and integrate only after a real
GPU run confirms both usable visual quality and RAM/VRAM headroom.

[Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) is another mesh candidate,
but upstream describes Windows support as experimental, requires native build tools, and its
weights are gated. Its [Stability AI Community License](https://stability.ai/news-updates/introducing-stable-fast-3d)
has eligibility conditions. Do not silently introduce an account requirement or treat it as
MIT-licensed. Evaluate it only if its terms and installation fit the product.

## Stage 3 — skeleton, skinning and correction

A useful humanoid needs semantically named bones and per-vertex skin weights. Predict a
skeleton and weights, show a few extreme test poses, and provide a correction workflow for
shoulders, elbows, hips, knees, hands and clothing. Clothing or hair fused to the body can
tear or deform badly; automated rigging cannot fix missing geometry or guarantee clean
weighting. Preserve the raw mesh and create a new rigged result.

[UniRig](https://github.com/VAST-AI-Research/UniRig) is an MIT-licensed research candidate
for skeleton and skinning prediction. Its upstream documentation specifies a CUDA GPU with
at least 8 GB VRAM, Python 3.11 and several native CUDA dependencies; these are requirements,
not proof that it fits this 8 GB Windows/16 GB RAM laptop. Its repository and
[model card](https://huggingface.co/VAST-AI/UniRig) are not fully consistent about which
skinning checkpoints are available; inspect the actual artifacts and licenses before selecting
one. [Make-It-Animatable](https://github.com/jasongzy/Make-It-Animatable) is another
MIT-licensed research candidate, but its inference memory and Windows setup need independent
verification. Neither should be advertised as a one-click VRM exporter without validation.

[AniGen](https://github.com/VAST-AI-Research/AniGen) produces articulated assets from one
image, but its documented minimum is 18 GB GPU memory, tested on Linux. It also includes a
third-party noncommercial component. It does not currently fit this product's target.

**Validation gate:** generated skeleton positions and weights are checked visually in
multiple poses, on several body types and layered outfits; test the actual Windows build,
peak GPU memory and free system RAM. Fail gracefully when the machine cannot run the engine.

## Stage 4 — face, hair, VRM export and real tracking

Create or fit face topology and expression blendshapes for separate eyelids and mouth shapes,
then map those shapes, humanoid bones, material information, look-at and spring bones into
VRM. Complex lashes, bangs, hats and layered skirts need clean geometry and appropriate
colliders; image-to-mesh and bone-only auto-rigging do not supply these automatically.
Provide manual adjustments and a preview before exporting. Author and usage information must
reflect the generated character and any assets used.

Validate actual exported VRM files in an independent VRM-compatible tool as well as Ka-myii,
including both eyes closing, mouth shapes, head turns and spring motion. Check available VRM
expressions rather than assuming a standard blendshape always exists. Face tracking should
hold or ease out when tracking is lost. Later, optional body and hand tracking can be tested
with supported devices and clear calibration; quality will depend on camera view, occlusion
and hardware. Professional studio and full-body tracking quality is a long-term target,
not a guarantee from one webcam.

## Next steps (after the first implementation)

- Open generated VRMs in VSeeFace / Unity UniVRM / VRoid Hub and fix whatever they reject.
- More characters: busts, long skirts, held props, chibi proportions; tune or fail clearly.
- Cap fused contacts instead of deleting triangles; finger bones from DWPose's hand points;
  eye bones for look-at; per-strand hair from See-through's hair layers.
- Better backs: a Redraw-from-behind sketch fed to Pixal3D's multi-view model (needs its
  5.6 GB pack).
- Draft / High quality in the UI; a progress thumbnail of the mesh.

## Release gates

- Local installation and offline rendering tested on the owner's Windows laptop; optional
  downloads, sizes and licensing are visible before use.
- User can reopen and export their own models without modifying source artwork or existing
  2D projects; bounded uploads and local-only resource loading follow existing security rules.
- Memory, cancellation and error cleanup tested for each GPU stage, with no concurrent GPU
  jobs and no hidden shared-memory spill.
- Browser and device checks distinguish synthetic input from a real webcam or microphone.
- Mesh quality, weight painting, facial expression fidelity and VRM interoperability are
  reviewed on real results, with missing capabilities reported rather than fabricated.
