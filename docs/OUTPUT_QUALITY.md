# Full-body output quality and next steps

The completion panel has Balanced (6.75-head target), Tall (7.5) and Compact (5.5) artistic
proportion targets. These are layout choices, not universal anatomy rules. Crown-to-chin hair
occupancy rejects sparse ahoge tips; a neck-centre axis aligns the retained design. A read-only
fit preview shows the actual planned placement and pose targets. Large headwear gets extra
space, and ankle targets leave room for shoes. Missing separated layers use approximate framing;
recent artwork-first extensions retain their measured geometry for later use.

The pose guide is an original OpenPose-format drawing generated from those measurements. A
pinned Apache-2.0 xinsir SDXL ControlNet conditions the extension locally, with no pose-detector
or OpenCV dependency. The guided path stores UNet/ControlNet weights in FP8 and computes in FP16,
encodes and discards text encoders before loading ControlNet, offloads to CPU and tiles VAE
decoding. The CUDA caching allocator is bounded for this operation and restored afterwards.
All models are released after the guided run; later jobs reload the checkpoint's original
precision. Original checkpoint files, source artwork and source artist-layer pixels are intact.

Masking alone was insufficient on the tested small retained head: some unguided trials generated
a second head below it. Initial differential masks now protect low-change pixels from step one,
source head alpha is hardened before protection, and a continuous lower-outfit repaint replaces
the old retained chest rectangle. Structural pose guidance is the recommended path for major
extensions. Generated anatomy, hands and outfit interpretation can still vary; the fit is a guide,
not a guarantee. Existing completed extensions need a new copy to use the revised method.

## Done since (2026-10-23, later)

- **Editable stance:** drag the shoulder, hip and ankle dots in the fit preview; each pair mirrors
  about the standing axis (level by design). Elbows, wrists and knees follow. Not draggable yet:
  elbows, asymmetric poses.
- **Output inspection:** after a full body is drawn: a second face (WD tagger on tiles down the
  body; the whole picture reads "solo" even with one), shoes/hat at the border, a tilted shoulder
  line (hair-coloured pixels ignored), tagger "bad hands". Each offers a fix (another seed, make
  room, repaint the shoulder box, refine hands). On 7 real full bodies: no false alarms; 3 of 4
  superseded alignment trials flagged for shoulders; 6/6 synthetic second heads caught.
- **Face-detail preservation:** separated extensions take the source's eyes, eyebrows, nose and
  mouth layers (scaled into place) where the new picture still shows them; painted expressions
  follow. Swapping the face or hair too left a grey forehead smudge, so those stay re-separated.
- **Refinement:** Refine hands / face redraws each part in its own close-up (crop context 15%).
- **Visemes:** painted A/I/U/E/O mouths with an LPC formant + pitch-scaled vowel classifier
  (30/30 nominal synthetic vowels, 85%+ with ±8% formant shifts, tested in Node).
- **Wardrobe/hair:** Alternate look repaints one part over the character without it and keeps
  only changed character pixels; the rig hides the replaced part while it is on.

- **Neck and size (2026-10-24):** the neck is redrawn from the jaw (no kept collar plate);
  bodies are sized from face width (1.33 face widths per head, measured on this checkpoint's own
  full bodies), shoulder joints ~1.4 face widths apart; a head-size check flags >15 % misses.

## Still open

1. Elbow/knee handles and asymmetric stances; a collar alignment check.
2. A licensed local upscaler with before/after review.
3. Variant hair physics tuned per style; deleting an alternate look from the UI.
4. Knee/hip anchors and arm-local deformers before waving or dance emotes.
5. Physical webcam/microphone trials (the vowel classifier is tested on synthetic vowels only)
   and a human Cubism import of a final package.

Reference study: [Live2D body movement](https://docs.live2d.com/en/cubism-editor-tutorials/deformer/),
[Jin Natori making video](https://docs.live2d.com/en/cubism-editor-tutorials/natori_making/),
[ControlNet model card](https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0),
[Diffusers memory techniques](https://huggingface.co/docs/diffusers/optimization/memory).
Reference videos are not redistributed as assets.
