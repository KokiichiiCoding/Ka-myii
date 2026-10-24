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
// A painted eye includes a broad skin plate; its alpha centroid is not the lid.
const painted=layer('eye_close',50,58,100,36);
for(let y=24;y<27;y++)for(let x=5;x<95;x++){
  const i=(y*100+x)*4;painted.imageData.data[i]=painted.imageData.data[i+1]=painted.imageData.data[i+2]=10;
}
for(let i=0;i<painted.imageData.data.length;i+=4){if(painted.imageData.data[i]===100){painted.imageData.data[i]=240;painted.imageData.data[i+1]=210;painted.imageData.data[i+2]=195;}}
for(let y=0;y<36;y++)for(let x=45;x<55;x++)painted.imageData.data[(y*100+x)*4+3]=0;
const paintedRig=R.buildRig({width:200,height:240,children:[layer('face',40,30,120,130),layer('eyewhite',55,65,25,20),layer('eyewhite',120,65,25,20),painted]});
assert.ok(Math.abs(paintedRig.anchors.eyeL.closeY-83)<1,'Anchor should follow dark lid ink, not the skin-patch centroid at y=76');
assert.equal(paintedRig.layers.find(l=>l.fade==='eyeClose').depth,paintedRig.layers.find(l=>l.name==='face').depth,'Skin plates and facial ink must share a head plane');
const cleanRig=R.buildRig({width:200,height:240,children:[layer('face',40,30,120,130),layer('eyewhite',55,65,25,20),layer('eyewhite',120,65,25,20),painted]},{simpleBlink:true,generic:{eyeL:GP.get('eyeL'),eyeR:GP.get('eyeR')}});
assert.ok(cleanRig.synth.eye);assert.equal(cleanRig.layers.filter(p=>p.fade==='eyeClose').length,2);
assert.ok(cleanRig.layers.filter(p=>p.fade==='eyeClose').every(p=>p.synthetic),'A clean fallback must replace both painted skin patches');
const eye={x0:55,x1:80,closeY:83};
assert.deepEqual(RT.closedEyePoint(67.5,83,eye,83,1,0,0),[67.5,83]);
assert.deepEqual(RT.closedEyePoint(67.5,83,{...eye,closeY:93},83,1,0,0),[67.5,93],'Dragging a close anchor must move the artwork');
assert.deepEqual(RT.closedEyePoint(67.5,83,eye,83,1.8,0,.4),[67.5,83],'Scaling/rotating a lid must not raise its anchor');
assert.deepEqual(RT.closedEyePoint(67.5,83,{...eye,closeX:77.5,closeY:78},83,1,0,0),[77.5,78],'Horizontal and vertical blink placement must move the art');
assert.deepEqual(RT.anchors({eyeLClose:{dx:7,dy:-9},eyeRClose:{dy:4}},200),{eyeLClose:{dx:7,dy:-9},eyeRClose:{dy:4}},'New XY and legacy vertical-only blink offsets must survive validation');
const binding=RT.strandBinding([{x:25,rootY:10,tipY:110},{x:75,rootY:10,tipY:110}],50,110);
const springs=[{stiff:{dx:30},soft:{dx:80}},{stiff:{dx:30},soft:{dx:80}}];
assert.equal(RT.strandShift({...binding,u:0},springs,.65,.4,true,18),0,'Hair roots stay attached');
const shift=RT.strandShift(binding,springs,.65,.4,true,18);
assert.ok(shift>0&&shift<18,'Bangs have bounded movement on rapid turns');
for(const hz of [10,30,60,240]){
 const spring={x:0,v:0};for(let frame=0;frame<hz*5;frame++)RT.spring(spring,20,70,9,1/hz);
 assert.ok(Math.abs(spring.x-20)<.001,'Spring must settle at '+hz+' fps');
}
const M=require(path.join(lib,'motion.js'));
assert.equal(R.normName('nekomimi'),'cat ears');assert.equal(R.normName('cat_ears'),'cat ears');
assert.equal(R.normName('nekomimi-right'),'cat ears');assert.equal(R.normName('cat-tail'),'tail');
assert.equal(R.normName('ears'),'ears','Human ears must not become cat ears');
const animalRig=R.buildRig({width:200,height:240,children:[layer('face',40,50,120,130),layer('nekomimi-left',30,10,35,35),layer('nekomimi-right',135,10,35,35)]});
assert.equal(animalRig.layers.filter(p=>p.name.startsWith('cat ears_')).length,2,'Named animal ears split into independent animated sides');
const toggleRig=R.buildRig({width:200,height:240,children:[layer('face',40,50,120,130),layer('toggle > < Eyes',60,90,80,20),layer('Toggle Sweat drop',140,40,15,25)]});
const toggles=toggleRig.layers.filter(p=>p.fade==='toggle');
assert.deepEqual(toggles.map(p=>[p.name,p.group]),[['toggle > < eyes_l','head'],['toggle > < eyes_r','head'],['toggle sweat drop','head']],'Eye toggles split into blink-aware sides while symbols remain one piece');
assert.ok(!toggleRig.warnings.some(w=>/toggle/.test(w)),'Toggle layers are known, not "unknown layer" warnings');
assert.ok(toggles.filter(p=>p.toggleEye).every(p=>p.toggleKey==='toggle > < eyes'&&p.toggleEye==='closed'));
const openToggles=R.buildRig({width:200,height:240,children:[layer('face',40,50,120,130),layer('toggle Heart eyes',60,90,80,20),layer('front hair',40,20,120,40)]}).layers;
assert.deepEqual(openToggles.filter(p=>p.toggleEye).map(p=>p.side),['L','R']);
const singleToggleRig=R.buildRig({width:200,height:240,children:[layer('face',40,50,120,130),layer('toggle Blush',60,90,80,20),layer('front hair',40,20,120,40)]});
assert.equal(openToggles.find(p=>p.name==='front hair').z,singleToggleRig.layers.find(p=>p.name==='front hair').z,'Splitting a toggle keeps following artist layer IDs stable');
for(let t=0;t<=32;t+=.01){const a=M.idlePose(t),b=M.idlePose(t+.001);
 assert.ok(Math.abs(a.shift)<=.450000001&&Math.abs(a.shift-b.shift)<.001,'Weight transfers are bounded and continuous');
 assert.ok(M.breath(t)>=0&&M.breath(t)<=1,'Breathing is bounded');}
assert.equal(M.idlePose(.1).shift,M.idlePose(.6).shift,'A natural weight shift dwells before returning');
// Vowel mouths from audio: a source-filter vowel (pulse train through formant resonators).
{
 const VS=require(path.join(lib,'visemes.js'));
 const synth=(f0,F,rate,n)=>{const x=new Float64Array(n);let ph=0;for(let i=0;i<n;i++){ph+=f0/rate;if(ph>=1)ph-=1;x[i]=ph<.05?1:0;}
  let y=x;for(const [f,bw] of [[F[0],80],[F[1],100],[2900,140]]){const r=Math.exp(-Math.PI*bw/rate),c=2*r*Math.cos(2*Math.PI*f/rate),d=-r*r,o=new Float64Array(n);for(let i=0;i<n;i++)o[i]=y[i]+c*(o[i-1]||0)+d*(o[i-2]||0);y=o;}
  const m=Math.max(...y.map(Math.abs));return Float32Array.from(y,v=>v/m);};
 const voices={male:[[100,120,150],{a:[730,1100],i:[290,2250],u:[330,1350],e:[460,1850],o:[460,820]}],
  female:[[200,230,260],{a:[880,1500],i:[340,2800],u:[390,1650],e:[540,2350],o:[520,950]}]};
 const top=w=>w&&Object.entries(w).sort((a,b)=>b[1]-a[1])[0][0];
 let hits=0,total=0;
 for(const [pitches,F] of Object.values(voices))for(const f0 of pitches)for(const v of VS.VOWELS){
  const w=VS.classify(synth(f0,F[v],48000,2048),48000);
  assert.equal(top(w),v,`Vowel ${v} at ${f0} Hz`);
  assert.ok(Math.abs(Object.values(w).reduce((a,b)=>a+b,0)-1)<1e-9,'Weights sum to 1');
  for(const j of [.92,1.08]){total++;if(top(VS.classify(synth(f0,F[v].map((f,k)=>f*(k?2-j:j)),48000,2048),48000))===v)hits++;}
 }
 assert.ok(hits/total>=.85,'Vowels survive ±8% formant shifts: '+hits+'/'+total);
 assert.equal(VS.classify(new Float32Array(2048),48000),null,'Silence has no vowel');
 assert.ok(Math.abs(VS.pitch(Array.from({length:512},(_,i)=>Math.sin(2*Math.PI*230*i/12000)),12000)-230)<5,'Pitch, not a subharmonic');
 assert.deepEqual(VS.fromShape(.3,.6,0),{a:0,i:1,u:0,e:0,o:0});assert.equal(top(VS.fromShape(.8,-.5,0)),'o');
 const b=VS.blend({a:1,i:0,u:0,e:0,o:0},{a:0,i:0,u:0,e:0,o:1},1/60);assert.ok(b.o>0&&b.o<1&&Math.abs(b.a+b.o-1)<1e-9,'Crossfade');
 {const a=VS.stackAlpha({a:.5,i:0,u:0,e:0,o:.5},['a','i','u','e','o']);
  assert.equal(a.a,1);assert.equal(a.o,.5);
  const shown=a.a*(1-a.o);assert.ok(Math.abs(shown-.5)<1e-12,'Over-compositing the stack gives the weighted mix, fully opaque');}
 const vrig=R.buildRig({width:200,height:240,children:[layer('face',40,50,120,130),layer('mouth',90,150,20,8),layer('mouth_a',85,145,30,20),layer('mouth_o',88,145,24,22)]});
 assert.deepEqual(vrig.layers.filter(p=>p.fade==='viseme').map(p=>p.name),['mouth_a','mouth_o'],'Vowel mouths are known rig parts');
 assert.ok(!vrig.warnings.some(w=>/mouth_[ao]/.test(w)));
}
// Alternate looks are toggles that replace a part (src/core/variants.py).
{
 const vr=R.buildRig({width:200,height:240,children:[layer('face',40,50,120,130),layer('front hair',40,20,120,40),layer('topwear',30,170,140,70),
  layer('toggle Hair: short bob',40,20,120,50),layer('toggle Top: red hoodie',30,170,140,70)]});
 const hair=vr.layers.find(p=>p.name.startsWith('toggle hair')),top=vr.layers.find(p=>p.name.startsWith('toggle top'));
 assert.ok(hair&&hair.fade==='toggle'&&hair.group==='head'&&hair.replaces.includes('front hair')&&hair.variantPart==='hair');
 assert.ok(top&&top.fade==='toggle'&&top.group==='body'&&top.replaces.includes('topwear'));
 assert.ok(!vr.warnings.some(w=>/toggle/.test(w)),'Alternate looks are known layers');
 const mouthEmote=R.buildRig({width:200,height:240,children:[layer('face',40,50,120,130),layer('mouth',90,150,20,8),layer('toggle Tongue out',85,145,30,20)]}).layers.find(p=>p.fade==='toggle');
 assert.ok(mouthEmote.replaces.includes('mouth_open')&&mouthEmote.replaces.includes('mouth_a')&&mouthEmote.variantPart==='mouth','Mouth emotes stand in for the talking mouth');
}
// Blink timing from Live2D's sample motions: shut in ~0.1 s, open in ~0.2 s, lids never reverse.
{
 let shut=null,open=null,prev=1,closing=true;
 for(let d=0;d<1;d+=.001){const v=M.blink(d,.04);if(v<0){open=d;break;}
  if(closing){assert.ok(v<=prev+1e-12,'Lids only close until shut');if(v===0){shut=d;closing=false;}}
  else assert.ok(v>=prev-1e-12,'Lids only open after the hold');prev=v;}
 assert.ok(shut>.08&&shut<=.1&&open-shut-.04>.16&&open-shut-.04<.2,'Close ~0.1 s, open ~0.2 s');
 assert.ok(M.blink(.05,.04)<.75,'Closing accelerates rather than drifting');
 const h=[];for(let t=0;t<60;t+=.05)h.push(M.idleHead(t));
 assert.ok(h.every(p=>Math.abs(p.x)<=.15&&Math.abs(p.y)<=.08&&Math.abs(p.z)<=.1),'Idle head stays subtle');
 assert.ok(Math.abs(M.idleHead(6.5345).x)<1e-9&&Math.abs(M.idleHead(6.5345).z)>.02,'Incommensurate periods: turn and tilt never loop together');
 assert.ok(M.TURN_TILT>0&&M.TURN_TILT<.5,'Head tilts towards the turn, as Live2D Angle X/Z do');
}
const fullGeometry={cx:100,shoulderY:100,hipY:250,footY:500,width:140,height:120};
assert.deepEqual(M.torsoPoint(100,500,500,fullGeometry,{roll:1,yaw:1,lift:1,shift:2}),[100,500]);
assert.ok(M.torsoPoint(100,250,250,fullGeometry,{roll:0,yaw:0,lift:0,shift:1})[0]>110,'Hips transfer weight above fixed feet');
let previous=-1;
for(let i=0;i<=100;i++){
 const m=M.mouth(i/100,0,0,.72);
 assert.ok(m.alpha>=previous&&m.alpha>=0&&m.alpha<=1);previous=m.alpha;
 assert.ok(m.height>=.08&&m.height<=1&&Number.isFinite(m.width));
}
assert.equal(M.mouth(0,0,0,.72).alpha,0);assert.equal(M.mouth(1,0,0,.72).alpha,1);
assert.ok(M.mouth(.1,0,0,.72).alpha>0,'Small syllables must not stay closed behind the old threshold');
assert.ok(M.mouth(1,0,1,.72).width<M.mouth(1,0,0,.72).width,'Rounded mouths narrow continuously');
for(const hz of [10,30,60,240]){
 let v=0;for(let f=0;f<hz;f++)v=M.response(v,1,1/hz,.65);
 assert.ok(v>.999,'Lip attack reaches the target at '+hz+' fps');
 for(let f=0;f<hz;f++)v=M.response(v,0,1/hz,.65);
 assert.ok(v<.001,'Lips settle closed without jitter at '+hz+' fps');
 for(const side of [-1,1])for(let t=0;t<10;t+=1/hz){
  assert.ok(M.tailPoint(0,0,200,40,side,t,1.8).every(v=>Math.abs(v)<1e-12),'Tail root must remain attached');
  const tip=M.tailPoint(0,1,200,40,side,t,1.8);
  assert.ok(tip.every(Number.isFinite)&&Math.hypot(...tip)<400,'Tail is finite and bounded');
  assert.ok(M.earPoint(0,0,60,90,side,t).every(v=>Math.abs(v)<1e-12),'Ear base must stay attached');
 }
}
// Reference-style nod: head leads quieter shoulders; every facial feature uses one transform.
for(const hz of [10,30,60,240]){
 const head=[],body=[];const vertical={x:0,v:0};
 for(let i=0;i<hz*4;i++){
  const pose=M.nodPose(i/hz,.85,2.6);
  head.push(pose.head);body.push(pose.body);
  assert.ok(pose.squash>.94&&pose.squash<=1);
  const faceGap=M.nodPoint(100,150,130,pose,1)-M.nodPoint(80,150,130,pose,1);
  assert.ok(Math.abs(faceGap-20*pose.squash)<1e-10,'Facial spacing shares a uniform compression without independent drifting');
  assert.ok(Math.abs(M.nodPoint(150,150,130,pose,0)-150-pose.body*130)<1e-10,'The torso does not inherit the full head dip');
  RT.spring(vertical,pose.head*130,46,7,1/hz);
  assert.ok(Number.isFinite(vertical.x)&&Math.abs(vertical.x)<40,'Vertical hair spring is bounded at '+hz+' fps');
 }
 const span=v=>Math.max(...v)-Math.min(...v);
 assert.ok(span(head)>span(body)*5,'Head-led motion is distinct from whole-model bobbing');
}
assert.deepEqual(M.nodPose(0,0,2.6),{head:0,body:-0,squash:1,pitch:-0,tilt:0});
// Head sway: half the nod tempo, full depth at each side, lingers there.
assert.ok(Math.abs(M.nodPose(.5,0,1,.75).tilt-.75)<1e-12&&Math.abs(M.nodPose(1.5,0,1,.75).tilt+.75)<1e-12);
assert.ok(M.nodPose(.3,0,1,1).tilt>.85,'Sway holds near the side rather than passing straight through');
// Head roll: ~16° at full tilt; the neck and collar bend from the chin down to still shoulders.
{
 const NP={cx:100,cy:200},chin=170,a=M.ROLL,cw=(x,y)=>M.collarWeight(x,y,NP,chin,20,30,.35);
 assert.ok(a>.26&&a<.3,'Full tilt is a visible head roll');
 const head=M.rollPoint(80,chin,NP,chin,a,1),neck=M.rollPoint(80,chin,NP,chin,a,cw(80,chin));
 assert.ok(Math.hypot(head[0]-neck[0],head[1]-neck[1])<1e-9,'Neck meets the jaw exactly');
 assert.deepEqual(M.rollPoint(80,230,NP,chin,a,cw(80,230)),[80,230],'Below the collar the torso stays put');
 assert.equal(cw(150,200),0,'The shoulders beside the neck stay put');
 const mid=M.rollPoint(100,185,NP,chin,a,cw(100,185));
 assert.ok(mid[0]>100&&mid[0]<head[0]+20,'Mid-neck leans part of the way');
 const collar=M.rollPoint(110,200,NP,chin,a,cw(110,200)),jaw=M.rollPoint(110,chin,NP,chin,a,1);
 assert.ok(Math.abs(collar[0]-110-.35*(jaw[0]-110))<1e-9,'A collar at the Neck pivot follows a third of the chin');
}
// Shoulders lift from the neck outwards and fade below the shoulder line; arms rise whole.
{
 const NP={cx:100,cy:200},lift=(x,y,rigid)=>M.shoulderLift(x,y,NP,20,60,50,-3,7,rigid);
 assert.equal(lift(110,190),0,'The neck and collar do not shrug');
 assert.equal(lift(30,195),-3);assert.equal(lift(170,195),7,'Each shoulder lifts on its own');
 assert.equal(lift(170,260),0,'A cropped chest stays on the canvas bottom');
 assert.equal(lift(170,260,true),7,'Arms rise with their shoulder');
 assert.ok(lift(170,225)>0&&lift(170,225)<7);
 assert.equal(M.turnShift(100,250,100,80,400,50,10),10);
 assert.ok(Math.abs(M.turnShift(180,250,100,80,400,50,10)-3)<1e-9,'The torso outline turns less than its centre');
 assert.equal(M.turnShift(100,450,100,80,400,50,10),0,'Legs stay planted');
 const crown={cx:100,cy:50};
 assert.equal(M.hatWeight(100,50,crown,40),0);assert.equal(M.hatWeight(130,50,crown,40),0,'The hat band stays on the head');
 assert.equal(M.hatWeight(100,-30,crown,40),1,'The tip trails fully');
}
{
 const NP={cx:100,cy:200},chin=170,a=M.ROLL;
 // Hanging hair below the chin moves with the head, not opposite like a rigid rotation would.
 const tip=M.rollPoint(100,400,NP,chin,a,1),chinMid=M.rollPoint(100,chin,NP,chin,a,1);
 assert.ok(Math.abs(tip[0]-chinMid[0])<1e-9&&Math.abs(tip[1]-400-(chinMid[1]-chin))<1e-9);
 const low=M.rollPoint(40,400,NP,chin,a,1,130),side=M.rollPoint(40,chin,NP,chin,a,1,130);
 assert.ok(Math.abs(low[0]-40-(side[0]-40))<1e-9&&low[1]===400,'Long hair slides with the head but its cut-off end never lifts');
 const top=M.rollPoint(100,40,NP,chin,a,1);
 assert.ok(Math.abs(Math.atan2(top[0]-100,200-top[1])-a)<1e-9,'Points above the chin rotate rigidly about the neck pivot');
}
{
 const w=10,h=8,data=new Uint8ClampedArray(w*h*4);
 for(let y=2;y<6;y++)for(let x=3;x<7;x++)data[(y*w+x)*4+3]=255;
 const pts=M.rowExtents({width:w,height:h,data},100,50,4);
 assert.deepEqual([Math.min(...pts.map(p=>p[0])),Math.max(...pts.map(p=>p[0])),Math.min(...pts.map(p=>p[1])),Math.max(...pts.map(p=>p[1]))],[103,107,52,56],'Outline hugs opaque pixels, not the layer box');
}
assert.ok(Math.abs(F.toParams({...F.DEFAULT_NEUTRAL,roll:-.2},null,null).az*M.ROLL-.2)<.01,'Tracked roll maps about 1:1 onto the rig');
assert.equal(RT.strandShift({...binding,u:0},springs,1,.8,true,8),0,'Vertical physics keeps hair roots attached');
assert.ok(Math.abs(RT.strandShift(binding,springs,3,2,true,8))<=8,'Vertical bangs are smoothly limited');
const torsoGeometry={cx:100,shoulderY:100,hipY:260,width:140,height:120};
for(const roll of [-1,-.4,0,.4,1])for(const yaw of [-1,0,1]){
 const pose={roll,yaw,lift:.4};
 assert.deepEqual(M.torsoPoint(100,300,300,torsoGeometry,pose),[100,300],'Feet below the hip pivot stay planted');
 const chest=M.torsoPoint(100,100,100,torsoGeometry,pose),head=M.torsoPoint(100,50,50,torsoGeometry,pose);
 assert.ok([...chest,...head].every(Number.isFinite));
 if(roll)assert.ok(Math.abs(chest[0]-100-yaw*140*.075)>1,'The upper torso follows an arc, not just a yaw slide');
}
for(let i=0;i<=10;i++)for(let j=0;j<=10;j++)for(let k=0;k<=10;k++){
 const w=M.mouthWeights(i/10,j/10,k/10,1);
 assert.ok(Object.values(w).every(v=>v>=0&&v<=1));assert.ok(Math.abs(w.source+w.grin+w.round-1)<1e-12,'Mouth variants form a continuous normalized blend');
}
assert.deepEqual(M.mouthWeights(1,1,0,1),{round:0,grin:1,source:0});
assert.deepEqual(M.mouthWeights(1,1,1,1),{round:1,grin:0,source:0});
assert.equal(M.mouthWeights(1,1,1,0).source,1,'Original only disables added mouth art');
const ink=layer('mouth_close',40,60,80,30);for(let i=0;i<ink.imageData.data.length;i+=4){ink.imageData.data[i]=240;ink.imageData.data[i+1]=200;ink.imageData.data[i+2]=180;}
for(let y=24;y<=26;y++)for(let x=25;x<=55;x++){const i=(y*80+x)*4;ink.imageData.data[i]=70;ink.imageData.data[i+1]=30;ink.imageData.data[i+2]=35;}
const mouthBox=M.mouthBox({anchors:{face:{x0:0,x1:200},mouth:{cx:80,cy:75,x0:40,x1:120}},layers:[{name:'mouth_close',x:40,y:60,w:80,h:30,img:ink.imageData}]});
assert.equal(mouthBox.cy,85,'Expression mouth placement follows ink, not the broad skin-patch centre');
assert.equal(mouthBox.cx,80);
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
