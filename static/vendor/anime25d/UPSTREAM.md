Anime2.5DRig by 852wa
https://github.com/852wa/Anime2.5DRig
Pinned commit: 7ddbd9943ea3152561b3dc8348fd752c850f3e95
MIT. Local changes: Ka-myii model adapter, offline-only devices, English UI (all user-visible text,
including status messages, tooltips, part labels and errors; the Japanese layer-name keywords in rigger.js
stay, they match artists' PSDs), reduced noise threshold for small facial features.
ag-psd: MIT, https://github.com/Agamnentzar/ag-psd
MediaPipe FaceMesh 0.4.1633559619: Google LLC, Apache-2.0; see lib/vendor/face_mesh/NOTICE.md.

2026-10-18 preview fixes: lid-ink anchors for painted closed-eye patches, a shared face plane,
closed-eye edits around the lid rather than the skin centroid, shorter blinks, bounded bang
springs, and a shared strand displacement field that transports baked hair shading within
the face texture (preserving alpha, facial ink and the artist PSD). Shadow follow is adjustable.

Manual blink placement: independent XY close anchors and tilt, held-eye comparison views,
on-stage dragging, saved adjustments, and a reversible generic-lid fallback for bad painted patches.

2026-10-23 motion pass: continuous small-mouth compression/blending and adjustable lip response/
rounding; stronger, bounded hair follow-through and shared model bounce; camera motion retains
85% of hair amplitude. motion.js provides GPU-free motion helpers and original canvas-drawn
optional cat ears/tail (MIT), hair-matched/preset palettes and rooted mesh deformation. Named
cat-ear PSD layers get independent twitching without animating human ears. Preview canvas
padding and renderer offsets keep fitted accessories visible; artist exports are unchanged.
Companion fit and motion parameters use the existing saved settings/undo system. Idle and
Physics now stop their respective motion when switched off.

Reference nod refinement (2026-10-23): opt-in Springy nod with depth/speed, head-led vertical
motion, small shared facial compression, neck blend and quieter shoulders. Phase accumulation
keeps speed changes continuous; camera nods use tracking rather than a repeating idle cycle.
Rooted, bounded vertical hair springs and shadow UV motion share one displacement field.

Neck and tilt (2026-10-23): head roll widened from 0.07 rad to 0.28 rad at |angleZ| = 1 and
face-features.js roll gain from 1.4 to 3.6 (about 1:1 tracking). Roll pivots on the neck pivot;
neck/neckwear blend from the chin (full) to the pivot (none); head points below the chin slide
with the chin line and their lift fades over one face height. The torso no longer rotates about
the neck; it leans slightly about the hips instead. Optional Head sway (tilt loop at half the
nod tempo). Canvas padding uses opaque row extents and ignores art already cut at the border.

Body, collar and hats (2026-10-23): body layers follow the head through `Motion.collarWeight`
(full at the chin, 35 % at the neck pivot, fading below it and sideways past the neck), so the
collar bends with the neck. `shoulderLift` (Shrug slider; breathing, tilt-side lift and Idle roll
scaled by Body motion) and `turnShift` (torso centre turns further than its outline; legs fixed)
deform body layers on a finer mesh. Headwear trails the crown on its own springs (`hatWeight`:
still within half a face height of the crown, full beyond one). Preview PSD v12 closes cloth cracks and
strand copies beneath opaque hair.

Torso and expressive mouths (2026-10-23): a shared curved torso parent replaces the tiny rigid
whole-model lean. Separate yaw/roll/lift springs follow the head, shoulder-to-hip falloff keeps
the lower body fixed, and hair springs include torso displacement. Padding includes torso
extremes for uncut head art and companions. `motion.js` adds original canvas-drawn toothy-smile
and O-mouth textures, visible-ink placement, normalized crossfades and jaw-drop deformation.
Original-only mode, new presets and controls use existing settings/undo. Preview PSD v13 heals
mouth sockets under the replacement artwork; artist assets and Cubism exports are untouched.
Research and design rationale: docs/MOTION_NOTES.md in the Ka-myii project.

Expression toggles (2026-10-23): a layer named `toggle <name>` (any case) is a head overlay with
`fade: 'toggle'`, hidden by default and on a face-fine mesh. Its visibility is the switch, so
settings, undo, saving and OBS sync carry it unchanged. The Expressions section lists the toggles
as buttons (text from the name), and Shift+1–9 flip them. Preview PSD v14 adds Ka-myii's painted
toggles (blush, tears, emote eyes); old settings files leave them off.

Fidelity refinement (2026-10-23): Natural sway uses quintic weight-transfer curves with dwell
periods and asymmetric breathing. A common hip-to-foot deformation keeps feet rooted; separate
strength controls and a preset use the existing save/undo system. Blink-aware eye toggles split
at the facial centre, group under one button, migrate older single-layer settings, and retain
stable following artist-layer IDs. Open-eye variants fade with their own eyelid; squeezed eyes
suppress the ordinary open/closed eye art. Competing eye designs are exclusive. Gloom and Sparkle
aura are original local manga artwork (the aura floats/pulses only with Idle).

Visemes, alternate looks and reference motion (2026-10-23, late): `lib/visemes.js` (new, MIT,
Ka-myii's own) estimates A/I/U/E/O from microphone audio (LPC formants, pitch-scaled references)
and gives `mouth_a … mouth_o` layers (`fade:'viseme'`) normalized stack alphas. `toggle Hair: …` /
`Top:` / `Bottom:` / `Headwear:` layers are alternate looks that hide the part they replace
(`replaces`, one per part); mouth emotes (tongue out, cat mouth, fang) replace the talking mouth.
New drawn symbols float with Idle. Motion: Live2D-sample blink timing and idle periods,
turn-coupled tilt, eye-led gaze shifts, idle saccades, turn dips/blinks and speech beats.
