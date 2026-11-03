import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { createCompanions } from './3d_companions.js?v=3';

const el = id => document.getElementById('d3-' + id);
const stage = el('stage');
const clamp = (x,a,b) => Math.max(a,Math.min(b,Number.isFinite(x) ? x : 0));
let renderer, camera, scene, controls, current=null, request=0, pose=null, idle=true, cameraOn=false;
let last=performance.now(), blinkAt=last + 2600, blinkStart=0;
let smooth={mouth:0,left:0,right:0};
let motion=window.Kamyii3DMotion.create();
let selectedMouth='aa';
let gestureName=null, gestureStart=0, lively=true, nextLively=performance.now()+14000;
// Microphone lip sync: level opens the mouth; vowel weights pick between painted A I U E O mouths.
let micOn=false, visemes={a:1,i:0,u:0,e:0,o:0}, visemeTarget=visemes;
const mic=window.RigDevices.createMic({onState(state,message){
  micOn=state==='on';
  el('mic').textContent=state==='loading' ? 'Starting microphone…' : micOn ? 'Microphone off' : 'Use microphone';
  setChip(el('mic'),micOn);
  if(state==='error')cameraStatus(message,true);
}});
const activeExpressions=new Set();
const companions=createCompanions();
// Ears and tail are remembered per character (files by name).
const companionKey=()=>current?.source?.id || el('model-name').textContent || 'file';
function showCompanions(fitted){
  el('companions-wrap').hidden=!fitted.canEars;
  const s=companions.settings;
  for(const [id,key] of [['ears','ear'],['tail','tail'],['fur','fur'],['ear-size','earSize'],['ear-lift','earLift'],['tail-size','tailSize']])el(id).value=s[key];
  el('tail').disabled=!fitted.canTail;el('tail-hint').hidden=fitted.canTail;
}

// Per-browser stage preferences (background, last avatar); the page works without storage.
const prefs=(()=>{try{return JSON.parse(localStorage.getItem('kamyii.3d')||'{}');}catch{return {};}})();
function savePrefs(){try{localStorage.setItem('kamyii.3d',JSON.stringify(prefs));}catch{}}
function setBackground(name){
  if(!el('backgrounds').querySelector(`[data-bg="${name}"]`))name='studio';
  stage.dataset.bg=name;prefs.bg=name;
  for(const b of el('backgrounds').children)setChip(b,b.dataset.bg===name);
}
// Chest springs of Ka-myii rigs (3+): 0 holds still, .5 is as rigged (and as VRM apps play it), 1 bounces most.
function chestJoints(vrm){return [...(vrm?.springBoneManager?.joints || [])].filter(j=>/^bust[LR]$/.test(j.bone.name));}
function chestBounce(vrm,amount){
  const joints=chestJoints(vrm),a=clamp(amount,0,1)*2,mix=(x,y,t)=>x+(y-x)*t;
  for(const j of joints){
    const rig=j.rigged ??= {stiffness:j.settings.stiffness,dragForce:j.settings.dragForce};
    j.settings.stiffness=a<=1 ? mix(4,rig.stiffness,a) : mix(rig.stiffness,rig.stiffness*.45,a-1);
    j.settings.dragForce=a<=1 ? mix(1,rig.dragForce,a) : mix(rig.dragForce,rig.dragForce*.4,a-1);
  }
  return joints.length;
}
function status(message, error=false) { el('status').textContent=message; el('status').classList.toggle('d3-error',error); }
function cameraStatus(message,error=false){el('camera-status').textContent=message;el('camera-status').classList.toggle('d3-error',error);}
function capability(isVrm){
  for(const name of ['camera','mic','turn','tilt','mouth','idle','lively','reset'])el(name).disabled=!isVrm;
  for(const button of el('gestures').children)button.disabled=!isVrm;
  el('calibrate').disabled=!isVrm || !cameraOn;
  el('fit').disabled=!current;el('save').disabled=!current;el('record').disabled=!current || typeof MediaRecorder==='undefined';
  for(const b of el('framing').children)b.disabled=!current || (b.dataset.frame!=='full' && !isVrm);el('turntable').disabled=!current;
  el('fullscreen').disabled=!current || !stage.requestFullscreen;
}
const tracking=window.Kamyii3DTracking.create({
  onPose(p){pose=p;},
  onState(state,message){
    if(state==='off' || state==='error'){cameraOn=false;pose=null;}
    else if(state==='on' || state==='tracking' || state==='no-face' || state==='calibrating')cameraOn=true;
    el('camera').textContent=cameraOn || state==='loading' ? 'Turn camera off' : 'Turn camera on';
    el('camera').dataset.starting=state==='loading'?'1':'';
    el('calibrate').disabled=!current?.vrm || !cameraOn;
    const labels={loading:'Starting camera…',on:'Camera on — looking for a face',tracking:'Face tracking on',
      'no-face':'No face detected — holding briefly',calibrating:'Look forward to calibrate…',off:'Camera off'};
    cameraStatus(message || labels[state] || state,state==='error');
  }
});

function dispose(object){if(object)VRMUtils.deepDispose(object);}
function validateGlb(bytes){
  const view=new DataView(bytes);
  if(bytes.byteLength<20 || view.getUint32(0,true)!==0x46546c67 || view.getUint32(4,true)!==2 || view.getUint32(8,true)!==bytes.byteLength)throw new Error('Please choose a valid VRM or binary GLB file.');
  const n=view.getUint32(12,true);
  if(view.getUint32(16,true)!==0x4e4f534a || n>bytes.byteLength-20)throw new Error('This file has an invalid 3D model header.');
  let json;
  try{json=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,20,n)).trim());}catch{throw new Error('This file has invalid 3D model data.');}
  for(const item of [...(json.buffers||[]),...(json.images||[])]) {
    if(typeof item.uri==='string' && !item.uri.startsWith('data:'))throw new Error('This file refers to external resources. Please use a self-contained VRM or GLB.');
  }
}
function fit(){
  if(!current)return;
  const box=new THREE.Box3().setFromObject(current.scene), size=box.getSize(new THREE.Vector3()), center=box.getCenter(new THREE.Vector3());
  const span=Math.max(size.x,size.y,size.z,.5);
  if(!Number.isFinite(span) || !Number.isFinite(center.y))throw new Error('This model has invalid dimensions.');
  controls.target.copy(center);camera.position.set(center.x,center.y,center.z+span*2.15);
  camera.near=Math.max(.001,span/1000);camera.far=Math.max(30,span*100);camera.updateProjectionMatrix();
  controls.minDistance=span*.35;controls.maxDistance=span*8;controls.update();
}
// Bust and face shots are sized by shoulder width, which works for full bodies and busts alike.
function frame(kind){
  const node=name=>current?.vrm?.humanoid?.getNormalizedBoneNode(name);
  const head=node('head'),left=node('leftUpperArm'),right=node('rightUpperArm');
  if(kind==='full' || !head){fit();return;}
  const p=head.getWorldPosition(new THREE.Vector3());
  const shoulders=left && right ? left.getWorldPosition(new THREE.Vector3()).distanceTo(right.getWorldPosition(new THREE.Vector3())) : (current.rig?.height || 1)*.22;
  // The head pivot sits low in the head (near the chin on Ka-myii rigs), so aim above it.
  const span=shoulders*(kind==='face' ? 1.6 : 2.7);
  const target=new THREE.Vector3(p.x,p.y+shoulders*(kind==='face' ? .55 : .05),p.z);
  controls.target.copy(target);camera.position.set(target.x,target.y,target.z+span*2.15);
  controls.minDistance=Math.min(controls.minDistance,span);controls.update();
}
function relaxArms(vrm){
  // T-pose imports are useful for rigging, but start the stage in a softer stance. Only
  // adjust a nearly horizontal upper arm; already lowered artist poses stay untouched.
  const humanoid=vrm.humanoid;
  if(!humanoid)return;
  vrm.update(0);vrm.scene.updateMatrixWorld(true);
  for(const side of ['left','right']){
    const upper=humanoid.getNormalizedBoneNode(side+'UpperArm');
    const lower=humanoid.getNormalizedBoneNode(side+'LowerArm');
    if(!upper || !lower)continue;
    const shoulder=new THREE.Vector3(),elbow=new THREE.Vector3();
    upper.getWorldPosition(shoulder);lower.getWorldPosition(elbow);
    const d=elbow.clone().sub(shoulder),len=d.length();
    if(len<1e-5 || d.y/len < -.45 || Math.abs(d.x)/len < .5)continue;
    const initial=upper.rotation.z;let best=initial,bestY=d.y;
    for(const offset of [-1.2,-.9,-.6,-.3,.3,.6,.9,1.2]){
      upper.rotation.z=initial+offset;vrm.scene.updateMatrixWorld(true);
      upper.getWorldPosition(shoulder);lower.getWorldPosition(elbow);
      const y=elbow.y-shoulder.y;
      if(y<bestY){best=initial+offset;bestY=y;}
    }
    upper.rotation.z=best;vrm.scene.updateMatrixWorld(true);
  }
  vrm.update(0);
}
function rigInfo(vrm){
  // Measured in world space: which way each arm rises and which way the spine bends
  // forward. VRM0 and VRM1 flip these, and artist rest poses differ.
  const humanoid=vrm.humanoid;if(!humanoid)return null;
  const node=name=>humanoid.getNormalizedBoneNode(name);
  const world=o=>{vrm.scene.updateMatrixWorld(true);return o.getWorldPosition(new THREE.Vector3());};
  const size=new THREE.Box3().setFromObject(vrm.scene).getSize(new THREE.Vector3());
  const info={arms:{},pitch:1,hips:node('hips'),hipsY:node('hips')?.position.y || 0,height:size.y || 1};
  for(const side of ['left','right']){
    const upper=node(side+'UpperArm'),lower=node(side+'LowerArm');
    if(!upper || !lower)continue;
    const elevation=()=>{const d=world(lower).sub(world(upper));return Math.asin(clamp(d.y/(d.length() || 1),-1,1));};
    const base=upper.rotation.z,before=elevation();
    upper.rotation.z=base+.2;const after=elevation();upper.rotation.z=base;
    info.arms[side]={upper,lower,base,lowerBase:lower.rotation.z,elev:before,up:after>before ? 1 : -1};
  }
  const bend=node('spine') || node('chest') || node('hips'),head=node('head');
  if(bend && head){
    const x=bend.rotation.x,before=world(head).z;bend.rotation.x=x+.3;
    info.pitch=world(head).z>before ? 1 : -1;bend.rotation.x=x;
  }
  vrm.scene.updateMatrixWorld(true);
  return info;
}
function playGesture(name){
  if(!current?.vrm || !window.Kamyii3DMotion.GESTURES.includes(name))return;
  gestureName=name;gestureStart=performance.now();
}
function applyGesture(g,head,chest,spine){
  const info=current.rig;if(!info)return;
  const breath=motion.breath || 0;
  for(const [side,arm] of Object.entries(info.arms)){
    const a=g?.arms[side],w=a?.w || 0;
    // Each inhale lifts the arm a touch (the shoulder rises); gestures take over smoothly.
    arm.upper.rotation.z=arm.base+arm.up*(((a ? a.elev : arm.elev)-arm.elev)*w+.035*breath*(1-w));
    arm.lower.rotation.z=arm.lowerBase+arm.up*(a ? a.bend : 0)*w;
  }
  // A bust (no legs to jump with, e.g. the 0.5 m witch) only bobs a little instead of floating up.
  if(info.hips)info.hips.position.y=info.hipsY+(g ? g.lift : 0)*info.height*(info.height<1 ? .35 : 1);
  if(chest || spine)(chest || spine).rotation.x-=info.pitch*.02*breath;  // the chest lifts back on the inhale
  if(!g)return;
  const bend=spine || chest;
  if(head){head.rotation.x+=info.pitch*g.head.x;head.rotation.y+=g.head.y;head.rotation.z+=info.pitch*g.head.z;}
  if(bend){bend.rotation.x+=info.pitch*g.spine.x;bend.rotation.y+=g.spine.y;bend.rotation.z+=info.pitch*g.spine.z;}
}
async function openFile(file){
  if(!file)return;
  if(file.size>128*1024*1024){request++;el('characters').value=current?.source?.id || '';status('Choose a VRM or GLB smaller than 128 MB.',true);return;}
  await openBytes(file.name,()=>file.arrayBuffer());
}
// A Ka-myii character's own avatar (Gallery "Make 3D avatar"), served by the local app.
async function openCharacter(id,label){
  if(!id)return;
  const url=`/api/generation/artifact/${encodeURIComponent(id)}/vrm`;
  await openBytes(label || 'Ka-myii avatar',async()=>{
    const r=await fetch(url);
    if(!r.ok)throw new Error(r.status===404 ? 'This character has no 3D avatar yet. Use Make 3D avatar in the Gallery.' : 'Could not load the avatar from Ka-myii.');
    return r.arrayBuffer();
  },{id,url});
}
async function openBytes(label,read,source=null){
  const token=++request; status('Opening ' + label + '…');
  try{
    const bytes=await read();if(token!==request)return;
    validateGlb(bytes);
    const manager=new THREE.LoadingManager();
    manager.setURLModifier(url=>{
      if(url.startsWith('blob:') || url.startsWith('data:'))return url;
      throw new Error('External resources are not supported. Choose a self-contained VRM or GLB.');
    });
    const loader=new GLTFLoader(manager);
    loader.register(parser=>new VRMLoaderPlugin(parser));
    const gltf=await loader.parseAsync(bytes,'');
    if(token!==request){dispose(gltf.scene);return;}
    if(!gltf.scene)throw new Error('The 3D file has no scene.');
    const vrm=gltf.userData.vrm || null;
    if(vrm)VRMUtils.rotateVRM0(vrm);
    const old=current; tracking.stop();pose=null;gestureName=null;current={scene:gltf.scene,vrm,source,rig:null};scene.add(current.scene);
    try{if(vrm){relaxArms(vrm);current.rig=rigInfo(vrm);}fit();}catch(e){scene.remove(current.scene);dispose(current.scene);current=old;throw e;}
    if(old){scene.remove(old.scene);dispose(old.scene);}
    smooth={mouth:0,left:0,right:0};
    motion=window.Kamyii3DMotion.create();
    el('placeholder').hidden=true;el('model-name').textContent=label;capability(!!vrm);
    el('characters').value=source?.id || '';
    if(source){el('download').href=source.url;el('download').hidden=false;prefs.last=source.id;savePrefs();}
    else{el('download').hidden=true;el('download').removeAttribute('href');}
    expressionChips(vrm);mouthShapes(vrm);
    el('bounce-wrap').hidden=!chestBounce(vrm,prefs.bounce ?? .5);
    showCompanions(companions.load(vrm,prefs.companions?.[companionKey()]));
    if(vrm && prefs.frame && prefs.frame!=='full')frame(prefs.frame);  // the shot chosen last time
    for(const o of el('framing').children)setChip(o,o.dataset.frame===(vrm && prefs.frame || 'full'));
    if(vrm){
      const em=vrm.expressionManager;
      const missing=[];
      if(!VOWELS.some(([,key])=>em?.getExpression(key)))missing.push('mouth');
      if(!em?.getExpression('blink') && !(em?.getExpression('blinkLeft') && em?.getExpression('blinkRight')))missing.push('blink');
      status('VRM ready. Orbit the view or turn on your camera.' + (missing.length ? ' This avatar does not have standard '+missing.join(' or ')+' expressions.' : ''));
    } else status('GLB ready for viewing. A VRM avatar with a humanoid rig is needed for face tracking.');
  }catch(err){if(token===request){el('characters').value=current?.source?.id || '';status(err.message || 'Could not open this 3D file.',true);}}
}
// Known Ka-myii painted face variants occupy the same area; keep one per group.
// Other imported VRM expressions remain independent and can be combined.
const DRIVEN=new Set(['blink','blinkLeft','blinkRight','aa','ih','ou','ee','oh','neutral','lookUp','lookDown','lookLeft','lookRight']);
// VRM0 joy/fun arrive as happy/relaxed and repaint the eyes, so they share the eye group.
const EYE_EMOTES=new Set(['heart eyes','star eyes','> < eyes','spiral eyes','happy','relaxed']);
const LABELS={happy:'Joy',angry:'Angry',sad:'Sorrow',relaxed:'Fun',surprised:'Surprised'};
const MOUTH_EMOTES=new Set(['tongue out','cat mouth','fang']);
const VOWELS=[['A','aa'],['I','ih'],['U','ou'],['E','ee'],['O','oh']];
function groupOf(name){const n=name.toLowerCase();return EYE_EMOTES.has(n)?EYE_EMOTES:MOUTH_EMOTES.has(n)?MOUTH_EMOTES:null;}
function setChip(chip,on){chip.setAttribute('aria-pressed',String(on));chip.classList.toggle('on',on);}
function expressionChips(vrm){
  const row=el('expressions');row.replaceChildren();activeExpressions.clear();
  const names=(vrm?.expressionManager?.expressions || []).map(e=>e.expressionName).filter(n=>!DRIVEN.has(n));
  for(const name of names){
    const chip=document.createElement('button');chip.type='button';chip.className='d3-btn secondary';
    chip.textContent=LABELS[name] || name;chip.dataset.expression=name;setChip(chip,false);
    if(row.children.length<9){const key=document.createElement('kbd');key.textContent='⇧'+(row.children.length+1);key.setAttribute('aria-hidden','true');chip.append(key);chip.title=`Shift+${row.children.length+1}`;}
    chip.onclick=()=>{
      const on=!activeExpressions.has(name),group=groupOf(name);
      if(on && group){for(const other of [...activeExpressions]){
        if(groupOf(other)===group){activeExpressions.delete(other);expression(other,0);
          const previous=[...row.children].find(button=>button.dataset.expression===other);if(previous)setChip(previous,false);}
      }}
      if(on)activeExpressions.add(name);else{activeExpressions.delete(name);expression(name,0);}
      setChip(chip,on);if(on && !group)expression(name,1);
    };
    row.append(chip);
  }
  el('expressions-wrap').hidden=!names.length;
}
function mouthShapes(vrm){
  const row=el('mouth-shapes');row.replaceChildren();
  const em=vrm?.expressionManager;
  const available=VOWELS.filter(([,key])=>em?.getExpression(key));
  selectedMouth=available[0]?.[1] || 'aa';
  for(const [label,key] of available){
    const button=document.createElement('button');button.type='button';button.className='d3-btn secondary';
    button.textContent=label;button.dataset.mouth=key;setChip(button,key===selectedMouth);
    button.onclick=()=>{selectedMouth=key;for(const b of row.children)setChip(b,b.dataset.mouth===key);};
    row.append(button);
  }
  el('mouth-shapes-wrap').hidden=available.length<2;
}
function expression(name,value){
  const manager=current?.vrm?.expressionManager;
  if(manager && manager.getExpression(name))manager.setValue(name,clamp(value,0,1));
}
function applyMotion(dt,now){
  const vrm=current?.vrm;if(!vrm)return;
  const manualTurn=Number(el('turn').value), manualTilt=Number(el('tilt').value), manualMouth=Number(el('mouth').value);
  // The camera bridge holds a brief dropout; once it clears the pose, return to idle
  // even while the camera remains enabled and is searching for a face.
  const ambientIdle=idle && !pose;
  // Spontaneous small gestures keep an idle avatar from looking frozen; mostly subtle ones.
  if(!ambientIdle || !lively || gestureName)nextLively=Math.max(nextLively,now+9000);
  else if(now>=nextLively){gestureName=['nod','nod','wave','hop'][Math.floor(Math.random()*4)];gestureStart=now;nextLively=now+16000+Math.random()*18000;}
  if(ambientIdle && now>=blinkAt){blinkStart=now;blinkAt=now+3000+Math.random()*2800;}
  const b=ambientIdle && blinkStart ? window.Kamyii3DMotion.blink(now-blinkStart) : 0;
  const target={x:pose ? clamp(pose.ay,-1,1)*.28 : 0,
    y:pose ? clamp(pose.ax,-1,1)*.45 : manualTurn*.45,
    z:pose ? -clamp(pose.az,-1,1)*.28 : manualTilt*.28,
    mouth:pose ? clamp(pose.mo,0,1) : manualMouth,
    left:pose ? 1-clamp(pose.eR,0,1) : b,
    right:pose ? 1-clamp(pose.eL,0,1) : b};
  const level=micOn ? mic.sample(1,.05) : 0;
  target.mouth=Math.max(target.mouth,level);
  if(micOn && level>.12){const f=mic.frame(),w=f && window.RigVisemes.classify(f.samples,f.rate);if(w)visemeTarget=w;}
  visemes=window.RigVisemes.blend(visemes,visemeTarget,dt,18);
  const follow=window.Kamyii3DMotion.follow;
  smooth.mouth=follow(smooth.mouth,target.mouth,dt,22,10);
  smooth.left=follow(smooth.left,target.left,dt,60,30);
  smooth.right=follow(smooth.right,target.right,dt,60,30);
  window.Kamyii3DMotion.step(motion,{x:target.x,y:target.y,z:target.z},dt,now/1000,ambientIdle);
  const humanoid=vrm.humanoid;
  const head=humanoid?.getNormalizedBoneNode('head');
  const chest=humanoid?.getNormalizedBoneNode('chest');
  const spine=humanoid?.getNormalizedBoneNode('spine');
  // The torso is an ancestor of the head. Subtract its rotation from the head's
  // local pose so follow-through does not amplify the intended head turn.
  const body=motion.body, headPose=motion.head;
  const chestShare=chest ? (spine ? .65 : 1) : 0, spineShare=spine ? (chest ? .35 : 1) : 0;
  if(chest)chest.rotation.set(body.x*chestShare,body.y*chestShare,body.z*chestShare);
  if(spine)spine.rotation.set(body.x*spineShare,body.y*spineShare,body.z*spineShare);
  const share=chestShare+spineShare;
  if(head)head.rotation.set(headPose.x-body.x*share,headPose.y-body.y*share,headPose.z-body.z*share);
  const g=gestureName ? window.Kamyii3DMotion.gesture(gestureName,Math.max(0,now-gestureStart)/1000) : null;
  if(!g)gestureName=null;
  applyGesture(g,head,chest,spine);
  const em=vrm.expressionManager;
  const squeeze=[...activeExpressions].some(name=>name.toLowerCase()==='> < eyes');
  if(em?.getExpression('blinkLeft') && em?.getExpression('blinkRight')){
    expression('blink',0);expression('blinkLeft',squeeze?0:smooth.left);expression('blinkRight',squeeze?0:smooth.right);
  }else expression('blink',squeeze?0:(smooth.left+smooth.right)/2);
  const mouthEmote=[...activeExpressions].some(name=>MOUTH_EMOTES.has(name.toLowerCase()));
  const painted=VOWELS.filter(([,key])=>em?.getExpression(key));
  if(micOn && painted.length>1){   // the voice picks the vowel; weights renormalised over painted mouths
    const vowel=key=>({aa:'a',ih:'i',ou:'u',ee:'e',oh:'o'})[key];
    const total=painted.reduce((sum,[,key])=>sum+visemes[vowel(key)],0) || 1;
    for(const [,key] of painted)expression(key,mouthEmote ? 0 : smooth.mouth*visemes[vowel(key)]/total);
  }else for(const [,key] of VOWELS)expression(key,!mouthEmote && key===selectedMouth ? smooth.mouth : 0);
  for(const name of activeExpressions){
    const lower=name.toLowerCase();
    if(EYE_EMOTES.has(lower))expression(name,lower==='> < eyes' ? 1 : 1-Math.max(smooth.left,smooth.right));
    else if(MOUTH_EMOTES.has(lower))expression(name,1);
  }
  vrm.update(dt);
  // Stiffness and damping alone still allow spring lag during rapid movement.
  // At zero, reset only our chest joints after simulation; hair keeps its physics.
  if((prefs.bounce ?? .5)===0)for(const joint of chestJoints(vrm))joint.reset();
  companions.update(dt,now,idle);
}
function render(now){
  requestAnimationFrame(render);
  const dt=clamp((now-last)/1000,0,.1);last=now;
  applyMotion(dt,now);controls.update();renderer.render(scene,camera);
  if(companions.wantsColour())companions.colourFrom(stagePixel);   // fur that matches the hair as drawn
}
const probe=document.createElement('canvas');probe.width=probe.height=1;
function stagePixel(point){
  const v=point.clone().project(camera),c=renderer.domElement,g=probe.getContext('2d',{willReadFrequently:true});
  if(Math.abs(v.x)>1 || Math.abs(v.y)>1)return null;
  g.clearRect(0,0,1,1);g.drawImage(c,Math.floor((v.x+1)/2*c.width),Math.floor((1-v.y)/2*c.height),1,1,0,0,1,1);
  return [...g.getImageData(0,0,1,1).data];
}
function init(){
  try{
    renderer=new THREE.WebGLRenderer({alpha:true,antialias:true,preserveDrawingBuffer:true});
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1,2));renderer.outputColorSpace=THREE.SRGBColorSpace;stage.prepend(renderer.domElement);
    scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(30,1,.01,100);camera.position.set(0,1,4);
    controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.target.set(0,1,0);controls.update();
    scene.add(new THREE.HemisphereLight(0xffffff,0x999aab,2));
    const light=new THREE.DirectionalLight(0xffffff,1.7);light.position.set(1,2,3);scene.add(light);
    const resize=()=>{const w=stage.clientWidth,h=stage.clientHeight;if(w&&h){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}};
    new ResizeObserver(resize).observe(stage);resize();requestAnimationFrame(render);
  }catch(e){status('3D preview needs WebGL. Try a current browser with hardware acceleration.',true);el('open').disabled=true;return;}
  el('open').onclick=()=>el('file').click();
  el('file').onchange=e=>{openFile(e.target.files[0]);e.target.value='';};
  stage.addEventListener('dragover',e=>{e.preventDefault();});
  stage.addEventListener('drop',e=>{e.preventDefault();openFile(e.dataTransfer.files[0]);});
  el('fit').onclick=fit;
  el('camera').onclick=async()=>{
    if(cameraOn || el('camera').dataset.starting){tracking.stop();return;}
    try{await tracking.start();}catch(e){cameraStatus(e.message,true);}
  };
  el('mic').onclick=async()=>{
    if(micOn){mic.stop();return;}
    try{await mic.start();}catch(e){cameraStatus(e.message,true);}
  };
  el('calibrate').onclick=async()=>{try{await tracking.calibrate();}catch(e){cameraStatus(e.message,true);}};
  for(const name of ['turn','tilt','mouth'])el(name).oninput=()=>{
    const n=Number(el(name).value);el(name+'-value').textContent=name==='mouth'?Math.round(n*100)+'%':n.toFixed(2);
  };
  for(const b of el('framing').children)b.onclick=()=>{frame(b.dataset.frame);prefs.frame=b.dataset.frame;savePrefs();for(const o of el('framing').children)setChip(o,o===b);};
  el('turntable').onclick=()=>{controls.autoRotate=!controls.autoRotate;controls.autoRotateSpeed=1.2;setChip(el('turntable'),controls.autoRotate);};
  el('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else stage.requestFullscreen().catch(()=>status('Fullscreen is not available here.',true));};
  document.addEventListener('fullscreenchange',()=>{el('fullscreen').textContent=document.fullscreenElement ? 'Exit fullscreen' : 'Fullscreen';});
  const showLively=()=>{el('lively').textContent=lively?'Little gestures on':'Little gestures off';el('lively').setAttribute('aria-pressed',String(lively));};
  const showIdle=()=>{el('idle').textContent=idle?'Idle on':'Idle off';el('idle').setAttribute('aria-pressed',String(idle));};
  // Remembered with the background and framing.
  if(prefs.lively===false){lively=false;showLively();}
  if(prefs.idle===false){idle=false;showIdle();}
  el('lively').onclick=()=>{lively=!lively;showLively();prefs.lively=lively;savePrefs();};
  el('idle').onclick=()=>{idle=!idle;showIdle();prefs.idle=idle;savePrefs();};
  const showBounce=()=>{el('bounce-value').textContent=Math.round(el('bounce').value*100)+'%';};
  el('bounce').value=prefs.bounce ?? .5;showBounce();
  el('bounce').oninput=()=>{showBounce();prefs.bounce=Number(el('bounce').value);savePrefs();chestBounce(current?.vrm,prefs.bounce);};
  for(const [id,key,number] of [['ears','ear'],['tail','tail'],['fur','fur'],['ear-size','earSize',1],['ear-lift','earLift',1],['tail-size','tailSize',1]]){
    el(id).addEventListener('input',()=>{
      companions.set({[key]:number ? Number(el(id).value) : el(id).value});
      prefs.companions={...prefs.companions,[companionKey()]:companions.settings};savePrefs();
    });
  }
  for(const button of el('gestures').children)button.onclick=()=>playGesture(button.dataset.gesture);
  document.addEventListener('keydown',e=>{
    if(e.ctrlKey || e.metaKey || e.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName))return;
    // Shift+1–9 switch expressions (as in Live Preview); M the microphone.
    if(e.shiftKey && /^Digit[1-9]$/.test(e.code)){const chip=el('expressions').children[Number(e.code.slice(5))-1];if(chip){e.preventDefault();chip.click();}return;}
    if(e.key.toLowerCase()==='m' && !el('mic').disabled){e.preventDefault();el('mic').click();return;}
    const button=el('gestures').children[Number(e.key)-1];
    if(button && !button.disabled){e.preventDefault();playGesture(button.dataset.gesture);}
  });
  el('reset').onclick=()=>{for(const name of ['turn','tilt','mouth']){el(name).value=0;el(name).dispatchEvent(new Event('input'));}if(controls)fit();};
  // A short clip for sharing: the canvas is transparent, so the chosen background is drawn into
  // the scene while recording (a green screen stays exactly #00b140 for keying).
  let recorder=null;
  el('record').onclick=()=>{
    if(recorder){recorder.stop();return;}
    const colour={studio:'#26302d',green:'#00b140',blue:'#0047bb',dark:'#101211',light:'#ebe7de',checker:'#30352f'}[stage.dataset.bg] || '#26302d';
    const type=['video/webm;codecs=vp9','video/webm'].find(t=>MediaRecorder.isTypeSupported(t));
    const chunks=[];
    const stream=renderer.domElement.captureStream(30);
    recorder=new MediaRecorder(stream,{mimeType:type,videoBitsPerSecond:8e6});
    // Change the background only once capture has begun: switching it in the same task as
    // captureStream() left the first recording on a page empty (no frames at all).
    requestAnimationFrame(()=>{if(recorder)scene.background=new THREE.Color(colour);});
    recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
    recorder.onstop=()=>{
      scene.background=null;recorder=null;stream.getTracks().forEach(t=>t.stop());el('record').textContent='Record clip';setChip(el('record'),false);
      const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(chunks,{type:'video/webm'}));
      a.download='kamyii-3d.webm';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),60000);  // a slow download still reads it
      status('Saved kamyii-3d.webm.');
    };
    recorder.start(250);el('record').textContent='Stop recording';setChip(el('record'),true);status('Recording… try a gesture (keys 1–6).');
    const mine=recorder;setTimeout(()=>{if(recorder===mine && mine.state==='recording')mine.stop();},15000);
  };
  el('save').onclick=()=>{renderer.domElement.toBlob(blob=>{if(!blob){status('Could not save this picture.',true);return;}const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='kamyii-3d.png';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);},'image/png');};
  window.addEventListener('pagehide',()=>{tracking.dispose();mic.stop(true);});
  for(const b of el('backgrounds').children)b.onclick=()=>{setBackground(b.dataset.bg);savePrefs();};
  setBackground(prefs.bg || 'studio');
  el('characters').onchange=()=>{const o=el('characters').selectedOptions[0];openCharacter(o.value,o.textContent);};
  loadCharacters();
}
async function loadCharacters(){
  try{
    const r=await fetch('/api/generation/list');const data=await r.json();
    const name=m=>m.metadata?.display_name || m.display_name || (m.name?.startsWith('model_') ? `Study ${m.name.slice(6).toUpperCase()}` : m.name || m.id.slice(0,8));
    const mine=(data.models||[]).filter(m=>m.metadata?.avatar3d);
    for(const m of mine){const o=document.createElement('option');o.value=m.id;o.textContent=name(m);el('characters').append(o);}
    el('characters').disabled=!mine.length;
    if(!mine.length)el('characters').options[0].textContent='No 3D avatars yet';
    el('make').hidden=!!mine.length || !(data.models||[]).some(m=>['completed','sketch'].includes(m.status));
    // An explicit ?model= wins; otherwise reopen the avatar viewed last, if it still exists.
    const wanted=new URLSearchParams(location.search).get('model') || (mine.some(m=>m.id===prefs.last) ? prefs.last : null);
    if(wanted){const o=[...el('characters').options].find(o=>o.value===wanted);el('characters').value=o ? wanted : '';openCharacter(wanted,o?.textContent);}
  }catch{el('characters').disabled=true;}
}
init();
// A compact diagnostic surface for local, GPU-free browser checks; no model data leaves the page.
window.Kamyii3D = {get state(){
  const head=current?.vrm?.humanoid?.getNormalizedBoneNode('head');
  const chest=current?.vrm?.humanoid?.getNormalizedBoneNode('chest');
  const spine=current?.vrm?.humanoid?.getNormalizedBoneNode('spine');
  const em=current?.vrm?.expressionManager;
  const expressions=Object.fromEntries((em?.expressions || []).map(e=>[e.expressionName,em.getValue(e.expressionName)]));
  const arm=current?.rig?.arms?.right;
  return {loaded:!!current,vrm:!!current?.vrm,cameraOn,micOn,gesture:gestureName,
    rightArmLift:arm ? arm.up*(arm.upper.rotation.z-arm.base) : 0,
    hipsLift:current?.rig?.hips ? current.rig.hips.position.y-current.rig.hipsY : 0,tracking:!!pose,idleActive:idle && !pose,name:el('model-name').textContent,
    headY:(head?.rotation.y || 0)+(chest?.rotation.y || 0)+(spine?.rotation.y || 0),
    chestY:chest?.rotation.y || 0,spineY:spine?.rotation.y || 0,
    mouthShape:selectedMouth,expressions,companions:companions.state(),
    chestSprings:chestJoints(current?.vrm).map(j=>({stiffness:j.settings.stiffness,drag:j.settings.dragForce,
      swing:2*Math.acos(Math.min(1,Math.abs(j.bone.quaternion.w)))}))};
}};
