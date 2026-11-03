const assert = require('node:assert/strict');
const motion = require('../static/js/3d_motion.js');
const target = {x:.2,y:.4,z:-.2};
function run(fps, seconds, idle=false, input=target) {
  const state=motion.create();
  for(let i=0;i<fps*seconds;i++) motion.step(state,input,1/fps,i/fps,idle);
  return state;
}
const reference=run(60,2);
for(const fps of [10,30,60,144,240]) {
  const state=run(fps,2);
  for(const axis of ['x','y','z']) {
    assert.ok(Math.abs(state.head[axis]-reference.head[axis])<.001, `head ${axis} stable at ${fps} fps`);
    assert.ok(Math.abs(state.body[axis]-reference.body[axis])<.006, `body ${axis} stable at ${fps} fps`);
    assert.ok(Math.abs(state.body[axis])<Math.abs(state.head[axis])*.25, 'body follows more gently than head');
  }
}
const first=run(60,1/60);
assert.ok(Math.abs(first.body.y)<Math.abs(first.head.y)*.05,'torso follows after the head');
const still=run(60,2,false,{x:0,y:0,z:0});
assert.deepEqual(still,motion.create(),'idle off is still at neutral');
const idle=run(60,2,true,{x:0,y:0,z:0});
assert.ok(Math.abs(idle.head.y)>.005 && Math.abs(idle.body.y)>.001,'idle moves head and torso');
const bounded=run(60,2,true,{x:999,y:-999,z:Infinity});
assert.ok(Math.abs(bounded.head.x)<=.28 && Math.abs(bounded.head.y)<=.45 && Number.isFinite(bounded.head.z));
for(const fps of [30,60,144]) {
  let value=0,max=0;
  for(let i=0;i<fps;i++) {
    value=motion.follow(value,motion.blink(i*1000/fps),1/fps,60,30);
    max=Math.max(max,value);
  }
  assert.ok(max>.95, `auto-blink closes fully at ${fps} fps`);
  assert.ok(value<.001,'blink reopens after the hold');
}
assert.equal(motion.blink(-1),0);
assert.equal(motion.blink(400),0);
const opened=motion.follow(0,1,.05,22,10);
const released=motion.follow(1,0,.05,22,10);
assert.ok(opened>.6 && released>.6,'mouth opens quickly and eases closed instead of snapping');
console.log('3D motion: bounded follow-through, frame-rate stability, idle and blink passed');
// Gestures start and end at the rest pose, so triggering one never snaps a bone.
for (const name of motion.GESTURES) {
  const duration = {wave:2.6,nod:1.25,shake:1.4,hop:1,bow:2.2,cheer:2}[name];
  assert.ok(duration, `${name} has a known duration`);
  for (const t of [0, duration - 1e-4]) {
    const g = motion.gesture(name, t);
    for (const arm of Object.values(g.arms)) assert.ok(arm.w < .02, `${name} arm weight ~0 at ${t}`);
    for (const v of [...Object.values(g.head), ...Object.values(g.spine), g.lift]) assert.ok(Math.abs(v) < .02, `${name} ~rest at ${t}`);
  }
  assert.equal(motion.gesture(name, duration), null, `${name} ends`);
}
assert.equal(motion.gesture('moonwalk', .5), null);
assert.equal(motion.gesture('wave', -1), null);
{
  const bends = [];
  for (let t = .6; t < 2; t += .02) { const arm = motion.gesture('wave', t).arms.right; assert.ok(arm.w > .99); bends.push(arm.bend); }
  assert.ok(Math.max(...bends) - Math.min(...bends) > .5, 'the forearm actually waves');
  assert.ok(Math.min(...bends) > 1, 'the hand stays raised while waving');
}
assert.ok(Math.max(...[.3,.4,.5].map(t => motion.gesture('hop', t).lift)) > .04, 'hop leaves the ground');
assert.ok(motion.gesture('bow', 1.1).spine.x > .3, 'bow bends forward');
console.log('3D gestures: seamless start/end, wave, hop and bow passed');
{
  const breathing = motion.create(), still = motion.create();
  let max = 0;
  for (let i = 0; i < 60 * 6; i++) { motion.step(breathing, {x:0,y:0,z:0}, 1/60, i/60, true); max = Math.max(max, breathing.breath); }
  assert.ok(max > .8 && max <= 1, 'idle breathes in fully within a few seconds');
  for (let i = 0; i < 60 * 3; i++) motion.step(breathing, {x:0,y:0,z:0}, 1/60, 6 + i/60, false);
  assert.ok(breathing.breath < .01, 'breathing settles when idle stops');
  motion.step(still, {x:0,y:0,z:0}, 1/60, 1, false);
  assert.equal(still.breath, 0);
  console.log('3D breathing passed');
}
