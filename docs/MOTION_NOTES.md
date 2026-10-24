# Preview movement notes

Research checked 2026-10-23; these are design references, not imported code or artwork.

- [Live2D: Motion Quality Improvement Tips](https://docs.live2d.com/en/cubism-editor-tutorials/motion-hint/)
  recommends coordinated face/body turns and lean, varied timing, inertia and recoil, and
  combining speech with other movement rather than moving the mouth alone.
- [Live2D: Body Movement](https://docs.live2d.com/en/cubism-editor-tutorials/deformer/)
  demonstrates nested body deformers, curved tilt and vertical movement with grounded feet.
- [Live2D: Standard Parameter List](https://docs.live2d.com/en/cubism-editor-manual/standard-parameter-list/)
  separates body turn/lean/breath from mouth form and opening.
- [Live2D: AIUEO mouth movement](https://docs.live2d.com/en/cubism-editor-tutorials/mouth-aiueo/)
  demonstrates distinct mouth shapes rather than scaling one open mouth for every expression.

## Choices in Ka-myii

These are our adaptations for the layered browser rig, not Live2D-prescribed numeric settings.
The torso is a common parent deformation for head, collar, clothing, arms and accessories.
Yaw, roll and lift have independent damped springs. A smooth shoulder-to-hip weight follows an
arc and reaches zero at the hips/canvas bottom, preventing the old floating lower edge. Idle
roll varies slowly; head turns and speech add smaller responses. Local neck/shoulder motion
remains on top, and hair inertia includes the torso displacement. Manual Body turn/Body tilt
and Body motion strength let an owner tune a quieter or more animated character.

Toothy smile and rounded O are original canvas artwork centered on the detected mouth ink,
with a fallback to the editable mouth anchor. Normalized blend weights fade out the source
mouth as a variant fades in. Jaw drop also stretches the chin smoothly. Old mouth holes are
healed only in the disposable preview PSD. Original-only mode retains the artist's painted
mouths. Settings save with the character; preview PNGs and recordings include these shapes.

## Limits

This remains a 2.5D deformation of existing pixels, without a back view or true 3D torso.
Camera input currently follows head/face motion; inferred body follow is not full-body tracking.
Mouth presets do not recognize spoken phonemes. Existing artwork/layer separation can still
limit the quality of large poses. Artist PSD, ORA and Cubism scaffolds are unchanged by browser
movement and expression additions. Physical webcam feel still needs owner testing.

## Further reference study and refinement

Reviewed [Live2D's Jin Natori making video](https://docs.live2d.com/en/cubism-editor-tutorials/natori_making/)
(body movement from 11:01, secondary movement from 11:47) and the body-deformer tutorial video.
The observed upper-body turn carries the collar/shoulders together; accessories settle separately.
Our adaptation adds slower weight transfers with a dwell rather than a constant sinusoidal sway,
a hip-to-foot falloff, and a longer exhale. These curves are our interpretation, not extracted
motion capture. The owner's previously supplied Nina GIF informs the opt-in Springy nod; its
fast nodding is kept separate from the quieter Natural sway preset. No reference artwork is
bundled. Browser tests verify actual meshes and the grounded lower edge, independently of the
pure curve tests.

Eye-expression layers now have separate left/right blink behavior, one grouped UI switch,
exclusive competing eye designs and settings migration. Squeezed eyes deliberately replace both
ordinary eye states. Gloom remains on the face; the original drawn Sparkle aura follows the head
and animates with Idle. None of this claims phoneme recognition or full-body camera tracking.

## Measured from Live2D's official samples (2026-10-23)

Read from the open sample code and motion data in
[CubismWebSamples](https://github.com/Live2D/CubismWebSamples) and
[CubismWebFramework](https://github.com/Live2D/CubismWebFramework) (Hiyori m01/m02/m05, Haru
idle, Natori, Mark), analysed with a script, not copied into Ka-myii. Sign convention from the
[standard parameter list](https://docs.live2d.com/en/cubism-editor-manual/standard-parameter-list/):
positive Angle X and Angle Z both point to the screen's right.

| Observation | Value | Ka-myii |
|---|---|---|
| Blink (motions) | close ~100 ms, open ~200 ms; framework default 0.10 / 0.05 / 0.15 s, next blink random 0–7 s | `Motion.blink`: 0.09 s ease-in close, 0.04 s hold, 0.18 s ease-out open |
| Idle head (sample app's "breath") | Angle X ±7.5° @ 6.5345 s, Y ±4° @ 3.5345 s, Z ±5° @ 5.5345 s, body X ±2° @ 15.5 s: incommensurate, never loops | `Motion.idleHead`: same periods at ~60% amplitude (2.5D art shows strain earlier) |
| Turn and tilt together | Angle Z follows Angle X, r = 0.70–0.93, Z swings ≥ X | `TURN_TILT` 0.3 in idle/mouse/random (the camera supplies real roll) |
| Body vs head | body Z 0.19–0.57 × head Z, near-simultaneous (body leads by 0–120 ms); drag-look: body X = head X / 3 | unchanged torso spring (yaw 0.65, roll 0.35 × head) is inside this range |
| Hair physics inputs | head 60% + body 40% (X and Z angle); skirt/ribbons body only; long chains bend more at the tip | unchanged: head hair already follows the head plus torso, body sway follows the torso |

Added from animation practice, not from the samples: the eyes reach a new gaze target ~130 ms
before the head (random look), idle micro-saccades (eyes rest, then dart, usually back to centre),
a small dip on fast head turns plus a blink on big ones (Williams, *The Animator's Survival Kit*),
and a little nod on each strong syllable while talking (Live2D motion tips: move with speech).
