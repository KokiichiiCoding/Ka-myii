/* GPU-free checks for the reused preview runtime and offline input lifecycle. */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const lib=path.resolve(__dirname,'../static/vendor/anime25d/lib');
const R=require(path.join(lib,'rigger.js')),RT=require(path.join(lib,'runtime.js')),F=require(path.join(lib,'face-features.js'));
assert.equal(R.normName('front_hair'),'front hair');
assert.equal(R.normName('mouth'),'mouth_open');
assert.equal(R.normName('eye_close'),'eye_close');
assert.ok(F.Tracker);
assert.equal(RT.safeRelativePath('../escape.psd','.psd'),null);
assert.equal(RT.safeRelativePath('https://example.com/model.psd','.psd'),null);
const GP=require(path.join(lib,'genericparts.js'));
function layer(name,x,y,w,h){const data=new Uint8ClampedArray(w*h*4);for(let i=0;i<data.length;i+=4){data[i]=100;data[i+3]=255;}return{name,left:x,top:y,imageData:{width:w,height:h,data}};}
const rig=R.buildRig({width:200,height:240,children:[layer('face',40,30,120,130),layer('eyewhite',55,65,25,20),layer('eyewhite',120,65,25,20),layer('mouth_open',90,120,20,10)]},{generic:{eyeL:GP.get('eyeL'),eyeR:GP.get('eyeR'),mouth:GP.get('mouth')}});
assert.ok(rig.anchors.eyeL&&rig.anchors.eyeR);
assert.equal(rig.synth.eye,true);assert.equal(rig.synth.mouth,true);
assert.equal(rig.layers.filter(l=>l.name.startsWith('eye_close')).length,2);
const source=fs.readFileSync(path.join(lib,'devices.js'),'utf8');
assert.ok(!source.includes('https://'),'Devices must never fetch a remote tracking fallback');
function scope(getUserMedia){const self={FaceMesh:function(){}};const s={self,navigator:{mediaDevices:{getUserMedia}},isSecureContext:true,setTimeout,clearTimeout,Promise,performance};vm.createContext(s);vm.runInContext(source,s);return self.RigDevices;}
(async()=>{
  const missing=scope(async()=>{throw Object.assign(new Error('absent'),{name:'NotFoundError'});});
  const camera=missing.createCamera();await assert.rejects(camera.start(),/No camera found/);assert.equal(camera.active,false);
  const denied=scope(async()=>{throw Object.assign(new Error('denied'),{name:'NotAllowedError'});});
  const mic=denied.createMic();await assert.rejects(mic.start(),/permission was denied/);assert.equal(mic.active,false);
  let grant,stopped=0;
  const late=scope(()=>new Promise(resolve=>{grant=resolve;}));
  const pending=late.createCamera();const start=pending.start();pending.stop();grant({getTracks:()=>[{stop(){stopped++;}}]});
  assert.equal(await start,false);assert.equal(stopped,1);assert.equal(pending.active,false);
  console.log('Preview runtime: aliases, rig anchors, generated expressions, offline assets, missing devices, denied permission and late stream cleanup passed.');
})().catch(err=>{console.error(err);process.exitCode=1;});
