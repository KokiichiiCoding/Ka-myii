const assert = require('node:assert/strict');
const Tracking = require('../static/js/3d_tracking.js');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const deferred = () => { let resolve; const promise = new Promise(r => resolve = r); return {promise,resolve}; };

function fixture(ensure = () => Promise.resolve()) {
  let callbacks, timer = 1000, stops = 0, resets = 0, started = 0, active = false;
  const poses = [], states = [];
  class Tracker {
    constructor(options) { assert.equal(options.linkEyes, false); this.samples = null; }
    reset() { resets++; }
    update(lm, time, aspect) {
      assert.equal(aspect, 4/3);
      if (this.samples) this.samples.push(lm);
      return lm;
    }
    startCalibration() { this.samples = []; }
    finishCalibration() { const ok = this.samples.length; this.samples = null; return ok ? {} : null; }
  }
  const devices = { createCamera(options) { callbacks = options; return {
    async start() { started++; active = true; callbacks.onState('on'); return true; },
    stop() { stops++; active = false; callbacks.onState('off'); }, get active() { return active; }
  }; }};
  const bridge = Tracking.create({devices, features:{Tracker}, ensureFaceMesh:ensure, now:() => timer,
    onPose:value => poses.push(value), onState:(s,m) => states.push([s,m])});
  return {bridge,poses,states,callbacks:()=>callbacks, advance:ms=>{timer+=ms;}, info:()=>({stops,resets,started})};
}

(async () => {
  const pending=deferred(), cancelled=fixture(() => pending.promise);
  const task=cancelled.bridge.start(); cancelled.bridge.stop(); pending.resolve();
  assert.equal(await task, false, 'Cancel before asset loading must not request a camera');
  assert.equal(cancelled.info().started, 0);

  const permission=deferred();let token=0, stoppedLateTrack=false;
  const lateDevice={createCamera(){return {
    async start(){const mine=++token;await permission.promise;if(mine!==token){stoppedLateTrack=true;return false;}return true;},
    stop(){token++;},get active(){return false;}
  };}};
  class MinimalTracker {constructor(){this.samples=null;}reset(){}}
  const late=Tracking.create({devices:lateDevice,features:{Tracker:MinimalTracker},ensureFaceMesh:async()=>{}});
  const lateStart=late.start();await Promise.resolve();await Promise.resolve();late.stop();permission.resolve();
  assert.equal(await lateStart,false);assert.equal(stoppedLateTrack,true,'Late permission stream is stopped by RigDevices');

  const f=fixture(); assert.equal(await f.bridge.start(), true);
  const face={ax:.3,ay:-.2,az:.1,eL:0,eR:1,mo:.5,ex:0,ey:0,br:0,mf:0};
  f.callbacks().onResults(face,{videoWidth:640,videoHeight:480});
  assert.equal(f.poses.at(-1),face,'Forward independent screen-side eye values unchanged');
  f.callbacks().onResults(null,{videoWidth:640,videoHeight:480});
  f.advance(1100);await sleep(220);
  assert.equal(f.poses.at(-1),face,'Brief face dropouts hold the last pose');
  f.advance(150);await sleep(220);
  assert.equal(f.poses.at(-1),null,'Long dropouts hand back to idle');
  assert.equal(f.states.at(-1)[0],'no-face');
  f.callbacks().onResults(face,{videoWidth:640,videoHeight:480});
  assert.equal(f.states.at(-1)[0],'tracking');
  const cal=f.bridge.calibrate();for(let i=0;i<5;i++)f.callbacks().onResults(face,{videoWidth:640,videoHeight:480});
  f.advance(1000);assert.equal(await cal,true,'Calibration stores a neutral pose from valid samples');
  const calCancelled=f.bridge.calibrate();f.bridge.dispose();
  assert.equal(await calCancelled,false,'Disposal cancels calibration');
  assert.equal(f.bridge.active,false);assert.equal(f.poses.at(-1),null);
  assert.ok(f.info().stops>=2 && f.info().resets>=2);
  console.log('3D camera bridge: cancellation, face dropout, calibration and cleanup passed');
})().catch(err=>{console.error(err);process.exitCode=1;});
