/* Small, bounded follow-through for a VRM's humanoid bones. Angles are radians. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Kamyii3DMotion = factory();
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';
  const clamp = (value, limit) => Math.max(-limit, Math.min(limit, Number.isFinite(value) ? value : 0));
  const blend = (rate, dt) => 1 - Math.exp(-rate * Math.max(0, Math.min(.1, dt || 0)));
  function follow(value, target, dt, attack, release) {
    return value + (target-value)*blend(target>value ? attack : release,dt);
  }
  function blink(ms) {
    // A short hold lets the eyes actually meet; smoothing a triangular blink at the
    // same speed as the head left many avatars visibly half-open.
    if (ms < 0 || ms >= 310) return 0;
    if (ms < 90) { const p=ms/90; return p*p*(3-2*p); }
    if (ms < 150) return 1;
    const p=(ms-150)/160; return 1-p*p*(3-2*p);
  }
  function create() { return {head:{x:0,y:0,z:0}, body:{x:0,y:0,z:0}, breath:0}; }
  function step(state, target, dt, seconds, idle) {
    // Idle is deliberately subdued and uses unrelated periods so it does not look like
    // a metronome. Camera and manual poses remain the main source of movement.
    const t = Number.isFinite(seconds) ? seconds : 0;
    const extra = idle ? {
      x: .014 * Math.sin(t * .87),
      y: .035 * Math.sin(t * .48) + .014 * Math.sin(t * .93),
      z: .018 * Math.sin(t * .61)
    } : {x:0,y:0,z:0};
    const limits = {x:.28,y:.45,z:.28};
    const headBlend = blend(12,dt), bodyBlend = blend(3.2,dt);
    for (const axis of ['x','y','z']) {
      const value = clamp((Number.isFinite(target[axis]) ? target[axis] : 0) + extra[axis],limits[axis]);
      state.head[axis] += (value-state.head[axis])*headBlend;
      // The shoulders follow after the head, while the hips and legs stay planted.
      const bodyTarget = state.head[axis] * ({x:.16,y:.22,z:.24}[axis]);
      state.body[axis] += (bodyTarget-state.body[axis])*bodyBlend;
    }
    // Breathing: ~15 breaths a minute, a slow inhale and a quicker exhale; eases out when idle stops.
    const inhale = idle ? Math.pow(.5 - .5*Math.cos(t * 1.6), 1.4) : 0;
    state.breath += (inhale-state.breath)*blend(6,dt);
    return state;
  }
  // One-shot body gestures. Arms are given as an elevation from horizontal (radians, up is
  // positive) plus an elbow bend, blended in by w; the stage maps them onto whatever rest
  // pose the avatar has. head/spine are added on top of follow-through; lift is hips
  // height as a share of the avatar's height.
  const ease = p => { p=Math.max(0,Math.min(1,p)); return p*p*(3-2*p); };
  const envelope = (t, attack, end, release) => ease(t/attack) * (1-ease((t-(end-release))/release));
  const GESTURES = {
    wave:{duration:2.6, pose(t){
      const w=envelope(t,.35,2.6,.45), swing=Math.sin(Math.max(0,t-.3)*2*Math.PI*2.1);
      return {arms:{right:{w,elev:-.15,bend:1.55+.32*swing*ease((t-.25)/.3)}},head:{x:0,y:-.06*w,z:.07*w},spine:{x:0,y:0,z:-.02*w},lift:0};
    }},
    nod:{duration:1.25, pose(t){
      const w=envelope(t,.12,1.25,.25);
      return {arms:{},head:{x:.1*w*(1-Math.cos(Math.min(t,1.1)*Math.PI*2/.55)),y:0,z:0},spine:{x:.03*w,y:0,z:0},lift:0};
    }},
    shake:{duration:1.4, pose(t){
      const w=envelope(t,.12,1.4,.3);
      return {arms:{},head:{x:.04*w,y:.3*w*Math.sin(t*Math.PI*2*2.1)*(1-t/1.8),z:0},spine:{x:0,y:0,z:0},lift:0};
    }},
    hop:{duration:1.0, pose(t){
      // Crouch, spring up, land, settle: a small parabola between two dips.
      const crouch=t<.18 ? -Math.sin(t/.18*Math.PI)*.012 : 0;
      const air=t>=.18 && t<.62 ? Math.sin((t-.18)/.44*Math.PI)*.045 : 0;
      const land=t>=.62 && t<.85 ? -Math.sin((t-.62)/.23*Math.PI)*.01 : 0;
      const w=envelope(t,.15,1,.3), up=air/.045;
      return {arms:{left:{w,elev:-.95+.55*up,bend:.25},right:{w,elev:-.95+.55*up,bend:.25}},
        head:{x:-.08*up,y:0,z:0},spine:{x:(crouch+land)*4,y:0,z:0},lift:crouch+air+land};
    }},
    bow:{duration:2.2, pose(t){
      const w=envelope(t,.5,2.2,.6);
      return {arms:{},head:{x:.18*w,y:0,z:0},spine:{x:.38*w,y:0,z:0},lift:0};
    }},
    cheer:{duration:2.0, pose(t){
      const w=envelope(t,.3,2.0,.45), pump=Math.abs(Math.sin(Math.max(0,t-.3)*Math.PI*2.4));
      return {arms:{left:{w,elev:.7+.15*pump,bend:.35},right:{w,elev:.7+.15*pump,bend:.35}},
        head:{x:-.1*w,y:0,z:0},spine:{x:-.04*w,y:0,z:0},lift:.012*pump*w};
    }}
  };
  function gesture(name, seconds) {
    const g=GESTURES[name];
    if (!g || !(seconds>=0) || seconds>=g.duration) return null;
    return g.pose(seconds);
  }
  return {create,step,follow,blink,gesture,GESTURES:Object.keys(GESTURES)};
});
