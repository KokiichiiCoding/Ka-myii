/*!
 * Anime2.5DRig — app.js
 * Editor UI, animation, input devices and OBS view.
 * Classic script (no modules) so that index.html also works from file://.
 * MIT License
 */
(function(){
'use strict';
const APP_VERSION='2.0.0';
const RT=window.RigRuntime, FF=window.FaceFeatures;
const QUERY=new URLSearchParams(location.search);
const KAMYII=window.KamyiiPreview;
const OBS_MODE=QUERY.get('obs')==='1', CAMERA_MODE=QUERY.get('cam')==='1';
if(OBS_MODE)document.body.classList.add('obs');
const $=id=>document.getElementById(id);
const clamp=(v,a,b)=>v<a?a:v>b?b:v;
function smooth(t){t=clamp(t,0,1);return t*t*(3-2*t);}
const round2=v=>Math.round(v*1000)/1000;
$('appVersion').textContent='v'+APP_VERSION.replace(/\.0$/,'');

// ---------- small helpers ----------
const statusEl=$('appStatus');let statusTimer=null;
function status(message,error=false){
  statusEl.textContent=message;statusEl.classList.toggle('error',error);statusEl.classList.remove('fade');
  clearTimeout(statusTimer);if(!error)statusTimer=setTimeout(()=>statusEl.classList.add('fade'),6000);
}
function download(blob,name){
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.style.display='none';
  document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),10000);
}
function storageGet(key){try{return localStorage.getItem(key);}catch(err){return null;}}
function storageSet(key,value){localStorage.setItem(key,value);}
function baseFileName(){return RT.safeFileName(modelName,'avatar');}

// ---------- WebGL ----------
const cv=$('cv');
const gl=cv.getContext('webgl',{alpha:true,antialias:true,premultipliedAlpha:true,preserveDrawingBuffer:false});
if(!gl){$('dropStatus').textContent='WebGL is unavailable. Enable browser hardware acceleration and reload.';return;}
let renderer;
try{renderer=window.RigRenderer.create(gl);}catch(err){$('dropStatus').textContent='Could not start the renderer: '+err.message;return;}
const EXPORT_BG={transparent:[0,0,0,0],green:[0,177/255,64/255,1],dark:[20/255,21/255,28/255,1]};

// ---------- model state (rebuilt per PSD) ----------
let modelId='',modelName='',kamyiiModel=false,currentRig=null,background='checker',paused=false,contextLost=false,lastFrame=null,capturePending=false,dirty=true;
let layers=[],A=null,baseAnchors=null,anchorOffsets={},CW=768,CH=768,FS=1,NP=null,BP=null,FC=null,CHEST=null,hasEyeClose2=false;
let renderBox={left:0,top:0,w:768,h:768,key:''};
let lastPsd=null,lastPre=null,anchorMode=false,anchorPreview=null,highlightLayer=null,blinkEditor=false,blinkView='both',rigSimpleBlink=false;
const bounce={x:0,v:0,dy:0};
const motion={x:{x:0,v:0},y:{x:0,v:0},earL:{x:0,v:0},earR:{x:0,v:0},tail:{x:0,v:0},hatX:{x:0,v:0},hatY:{x:0,v:0},torsoYaw:{x:0,v:0},torsoRoll:{x:0,v:0},torsoLift:{x:0,v:0},torsoShift:{x:0,v:0},nodTime:0,palette:-1};
const Motion=window.RigMotion;
function invalidate(){dirty=true;}

function makeThumb(img){
  if(OBS_MODE||typeof document==='undefined')return '';
  try{
    const s=Math.min(1,64/Math.max(img.width,img.height)),tw=Math.max(1,Math.round(img.width*s)),th=Math.max(1,Math.round(img.height*s));
    const full=document.createElement('canvas');full.width=img.width;full.height=img.height;full.getContext('2d').putImageData(img,0,0);
    const t=document.createElement('canvas');t.width=tw;t.height=th;const ctx=t.getContext('2d');ctx.imageSmoothingQuality='high';ctx.drawImage(full,0,0,tw,th);
    return t.toDataURL('image/png');
  }catch(err){return '';}
}
function prepareLayers(rig){
  const prepared=[],W=rig.canvas.w;
  const maxTexture=gl.getParameter(gl.MAX_TEXTURE_SIZE),maxViewport=gl.getParameter(gl.MAX_VIEWPORT_DIMS);
  if(rig.canvas.w>maxViewport[0]||rig.canvas.h>maxViewport[1]||rig.canvas.w>maxTexture||rig.canvas.h>maxTexture)throw new Error('The canvas is larger than this device can draw ('+Math.min(maxViewport[0],maxTexture)+' px). Use a smaller PSD');
  try{
    for(const Lr of [...Motion.parts(rig),...rig.layers].sort((a,b)=>a.z-b.z)){
      if(Lr.w>maxTexture||Lr.h>maxTexture)throw new Error('An image part is larger than this device\'s limit of '+maxTexture+' px');
      const L=Object.assign({visible:Lr.fade!=='toggle',opacity:1},Lr,{id:String(Lr.z)+':'+Lr.name});
      delete L.img;
      L.defaultDepth=L.depth;L.defaultOpacity=L.opacity;
      prepared.push(L);
      const cell=(L.companion?10:L.name==='face'||L.fade==='eyeClose'||L.fade==='eyeClose2'||L.fade==='toggle'?8:L.phys||L.group==='body'?24:42)*Math.max(0.6,W/768);   // the collar and shoulders bend body layers
      const {nx,ny}=RT.meshSize(L.w,L.h,cell),nv=(nx+1)*(ny+1);
      const base=new Float32Array(nv*2),uv=new Float32Array(nv*2);
      let k=0;
      for(let j=0;j<=ny;j++)for(let i=0;i<=nx;i++){base[k]=L.x+L.w*i/nx;base[k+1]=L.y+L.h*j/ny;uv[k]=i/nx;uv[k+1]=j/ny;k+=2;}
      const idx=new Uint16Array(nx*ny*6);let q=0;
      for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){const a=j*(nx+1)+i,b=a+1,c=a+nx+1,d=c+1;idx[q++]=a;idx[q++]=b;idx[q++]=c;idx[q++]=b;idx[q++]=d;idx[q++]=c;}
      L.base=base;L.cur=new Float32Array(base);
      L.bn=L.expression?'mouth_open':Rigger.baseName(L.name.replace(/_(l|r)$/, ''));
      if(L.strands&&L.strands.length){
        const S=L.strands,nS=S.length;
        let spacing=120;
        if(nS>1){const ds=[];for(let s=1;s<nS;s++)ds.push(S[s].x-S[s-1].x);ds.sort((a,b)=>a-b);spacing=ds[ds.length>>1];}
        const sig=Math.max(1,spacing*0.6);
        L.sw=new Float32Array(nv*nS);L.su=new Float32Array(nv);
        L.spr=S.map((s,i)=>({stiff:{x:0,v:0,dx:0},soft:{x:0,v:0,dx:0},liftStiff:{x:0,v:0,dx:0},liftSoft:{x:0,v:0,dx:0},phase:i*1.37+L.z}));
        for(let v=0;v<nv;v++){
          const x=base[v*2],y=base[v*2+1];let tot=0;
          for(let s=0;s<nS;s++){const w=Math.exp(-Math.pow((x-S[s].x)/sig,2));L.sw[v*nS+s]=w;tot+=w;}
          let rY=0,tY=0;
          if(tot>1e-6){for(let s=0;s<nS;s++){L.sw[v*nS+s]/=tot;rY+=L.sw[v*nS+s]*S[s].rootY;tY+=L.sw[v*nS+s]*S[s].tipY;}}
          else{L.sw[v*nS]=1;rY=S[0].rootY;tY=S[0].tipY;}
          L.su[v]=clamp((y-rY)/Math.max(1,tY-rY),0,1);
        }
      }
      if(L.spr)L.liftSpr=L.spr.map(sp=>({stiff:sp.liftStiff,soft:sp.liftSoft}));
      if(L.su)L.bindings=Array.from(L.su,(u,i)=>({u,weights:L.sw.subarray(i*L.strands.length,(i+1)*L.strands.length)}));
      if(L.group!=='body'&&!L.companion)L.outline=Motion.rowExtents(Lr.img,L.x,L.y,32);
      const image=(typeof ImageData!=='undefined')?new ImageData(Lr.img.data,Lr.img.width,Lr.img.height):Lr.img;
      L.thumb=makeThumb(image);
      renderer.upload(L,{positions:L.cur,uvs:uv,indices:idx,image});
    }
    bindHairShading(prepared,rig);
    return prepared;
  }catch(err){prepared.forEach(renderer.dispose);throw err;}
}
// Skin shading near bangs is painted into the face by decomposition. Advect its
// texture with the same strand field as the hair, keeping facial ink and alpha fixed.
function bindHairShading(prepared,rig){
  const hairs=prepared.filter(L=>L.bn==='front hair'&&L.strands?.length);
  const fs=rig.anchors.faceScale,radius=Math.max(8,24*fs);
  function alpha(img,L,x,y){x=Math.round(x-L.x);y=Math.round(y-L.y);return x>=0&&y>=0&&x<img.width&&y<img.height?img.data[(y*img.width+x)*4+3]/255:0;}
  function outsideBox(x,y,b,pad){
    const dx=Math.max(b.x0-x,x-b.x1,0),dy=Math.max(b.y0-y,y-b.y1,0);
    return smooth(Math.hypot(dx,dy)/pad);
  }
  for(const skin of prepared.filter(L=>L.bn==='face'||L.fade==='eyeClose'||L.fade==='eyeClose2')){
    const src=rig.layers.find(L=>L.z===skin.z),bindings=[];
    skin.flow=new Float32Array(skin.base.length);
    for(let k=0;k<skin.base.length;k+=2){
      const x=skin.base[k],y=skin.base[k+1];let protection=1,best=null;
      // Do not transport the eye sockets, brows, nose or outline with the hair.
      for(const side of ['L','R']){
        const eye=rig.anchors['eye'+side];if(!eye)continue;
        const closed=skin.fade==='eyeClose'||skin.fade==='eyeClose2';
        protection*=outsideBox(x,y,{x0:eye.x0-4*fs,x1:eye.x1+4*fs,y0:closed?eye.closeY-3*fs:eye.y0-6*fs,y1:closed?eye.closeY+3*fs:eye.y1+4*fs},8*fs);
      }
      for(const part of rig.layers){const bn=Rigger.baseName(part.name.replace(/_(l|r)$/,''));
        if(bn==='eyebrow'||bn==='nose'||bn==='mouth_open'||bn==='mouth_close')protection*=outsideBox(x,y,{x0:part.x,x1:part.x+part.w,y0:part.y,y1:part.y+part.h},6*fs);
      }
      if(protection<.001||alpha(src.img,src,x,y)<.98)continue;
      // A zero-flow rim protects the cutout edge even under large movement.
      let rim=1;
      for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]])rim=Math.min(rim,alpha(src.img,src,x+dx*radius*.65,y+dy*radius*.65));
      if(rim<.98)continue;
      for(const hair of hairs){
        const art=rig.layers.find(L=>L.z===hair.z);let coverage=0;
        for(let dy=-radius;dy<=radius;dy+=radius/3)for(let dx=-radius;dx<=radius;dx+=radius/3){
          const weight=Math.max(0,1-Math.hypot(dx,dy)/radius);
          coverage=Math.max(coverage,alpha(art.img,art,x+dx,y+dy)*weight);
        }
        const weight=smooth(coverage)*protection;
        if(weight>.001&&(!best||weight>best.weight))best={k,hair,weight,x,y,binding:RT.strandBinding(hair.strands,x,y)};
      }
      if(best)bindings.push(best);
    }
    skin.shadingBindings=bindings;
  }
}
function hairShift(L,binding,e){
  const front=!!FRONTISH[L.bn],accessory=L.phys==='sway';
  const dx=auto.phys?RT.strandShift(binding,L.spr,accessory?e.accAmp*1.4:front?e.fhAmp:e.physAmp,accessory?.8:front?e.fhSoft:e.soft,front,front?(18+4*e.motionBounce)*FS:60*FS,accessory?1.2:undefined):0;
  return dx;
}
function hairLift(L,binding,e){
  if(!auto.phys||!L.liftSpr)return 0;
  const front=!!FRONTISH[L.bn],accessory=L.phys==='sway';
  return RT.strandShift(binding,L.liftSpr,accessory?e.accAmp*.6:front?e.fhAmp:e.physAmp,
    accessory?.7:front?e.fhSoft:e.soft,front,(front?8:26)*FS,accessory?1.2:undefined);
}
function bangShift(L,x,u,e){
  if(L.bn!=='front hair')return 0;
  const fw=A.face.x1-A.face.x0,fc=A.face.cx,f=36,s1=smooth((x-(fc-fw*.22))/f+.5),s2=smooth((x-(fc+fw*.22))/f+.5);
  return (e.bangL*(1-s1)+e.bangC*s1*(1-s2)+e.bangR*s2)*Math.pow(u,1.4)*22*FS;
}
function moveHairShading(L,e){
  if(!L.flow)return;
  L.flow.fill(0);
  for(const b of L.shadingBindings){
    if(!b.hair.visible||b.hair.opacity<.01)continue;
    const dx=hairShift(b.hair,b.binding,e)+bangShift(b.hair,b.x,b.binding.u,e);
    // The hair's additional parallax also moves its painted shadow across skin.
    const dd=b.hair.depth-L.depth,scale=b.weight*b.hair.opacity*e.shadowFollow;
    L.flow[b.k]=(dx+FS*e.angleX*40*dd)*scale/L.w;
    L.flow[b.k+1]=(hairLift(b.hair,b.binding,e)+Math.abs(dx)*.12-FS*e.angleY*dd*(30+(b.y-FC.y)*.05))*scale/L.h;
  }
  renderer.flow(L,L.flow);
}
// Front-hair block weights depend on the face anchor, so they follow anchor edits.
function buildBangWeights(L){
  if(L.bn!=='front hair'||!L.su){L.bw=null;return;}
  const fw=A.face.x1-A.face.x0,fcx=A.face.cx,f=36,b1=fcx-fw*0.22,b2=fcx+fw*0.22,nv=L.base.length/2;
  L.bw=new Float32Array(nv*3);
  for(let v=0;v<nv;v++){const x=L.base[v*2],s1=smooth((x-b1)/f+0.5),s2=smooth((x-b2)/f+0.5);L.bw[v*3]=1-s1;L.bw[v*3+1]=s1*(1-s2);L.bw[v*3+2]=s2;}
}
function applyAnchors(){
  if(!baseAnchors)return;
  const a=JSON.parse(JSON.stringify(baseAnchors)),o=anchorOffsets,off=k=>o[k]||{};
  a.face.cx+=off('face').dx||0;a.face.cy+=off('face').dy||0;
  for(const s of ['L','R']){const e=a['eye'+s];if(!e)continue;e.icx+=off('eye'+s).dx||0;e.icy+=off('eye'+s).dy||0;e.closeY+=off('eye'+s+'Close').dy||0;e.closeX=(e.x0+e.x1)/2+(off('eye'+s+'Close').dx||0);}
  const m=off('mouth');for(const k of ['cx','x0','x1'])a.mouth[k]+=m.dx||0;for(const k of ['cy','y0','y1'])a.mouth[k]+=m.dy||0;
  a.neckPivot.cx+=off('neck').dx||0;a.neckPivot.cy+=off('neck').dy||0;
  A=a;FS=A.faceScale;NP=A.neckPivot;BP=A.bodyPivot;FC={x:A.face.cx,y:A.face.cy};
  const ch=off('chest'),fh=A.face.y1-A.face.y0;
  CHEST={cx:NP.cx+(ch.dx||0),cy:A.neckBottom+fh*0.60+(ch.dy||0),rx:Math.max(1,(A.face.x1-A.face.x0)*0.60),ry:Math.max(1,fh*0.45)};
  for(const L of layers)buildBangWeights(L);
  const n=Object.keys(anchorOffsets).length;
  $('anchorSummary').textContent=n?n+' manually adjusted anchors':'Using automatic anchor positions';
  syncBlinkEditor();renderOverlay();invalidate();
}
function applyRig(rig){
  if(contextLost)throw new Error('Wait for the renderer to recover, then load again');
  blinkEditor=false;anchorMode=false;anchorPreview=null;
  const prepared=prepareLayers(rig);
  layers.forEach(renderer.dispose);layers=prepared;currentRig=rig;
  CW=rig.canvas.w;CH=rig.canvas.h;renderBox={left:0,top:0,w:CW,h:CH,key:''};
  baseAnchors=JSON.parse(JSON.stringify(rig.anchors));anchorOffsets={};
  applyAnchors();
  Object.assign(bounce,{x:0,v:0,dy:0});
  for(const k of ['x','y','earL','earR','tail','hatX','hatY','torsoYaw','torsoRoll','torsoLift','torsoShift'])Object.assign(motion[k],{x:0,v:0});motion.palette=-1;motion.nodTime=0;lastFrame=null;blinkT=-1;blinkVariant=1;irisBounceT=-1;
  hasEyeClose2=layers.some(L=>L.bn==='eye_close2');maskFailed=false;
  visemeOrder=layers.filter(L=>L.fade==='viseme').sort((a,b)=>a.z-b.z).map(L=>L.bn.slice(6));visemeSet=new Set(visemeOrder);visemes=oneVowel('a');visemeTarget=oneVowel('a');
  cv.width=CW;cv.height=CH;
  $('overlay').setAttribute('viewBox','0 0 '+CW+' '+CH);
  resetView();fit();
  expandedLayers.clear();renderLayerList();renderDiagnostics(rig);
  $('drop').classList.add('hidden');
  document.querySelectorAll('[data-model-action]').forEach(b=>b.disabled=false);
  updateRecordAvailability();
  invalidate();
}

// ---------- PSD loading ----------
const dropEl=$('drop'),dropStatus=$('dropStatus');
let loadTicket=0,loaderWorker=null,loadController=null,cancelWorker=null;
function setLoading(on){$('btnCancelLoad').hidden=!on;$('loadProgress').hidden=!on;}
function cancelLoad(){
  loadTicket++;loadController?.abort();loadController=null;
  loaderWorker?.terminate();loaderWorker=null;
  cancelWorker?.(new DOMException('Cancelled','AbortError'));cancelWorker=null;
  setLoading(false);
  dropEl.classList.toggle('hidden',!!layers.length);dropStatus.textContent='';
}
function loadProgress(message){dropStatus.textContent=message;status(message);}
async function parseModel(buffer,generic,ticket){
  if(location.protocol!=='file:'&&typeof Worker!=='undefined'){
    return new Promise((resolve,reject)=>{
      cancelWorker=reject;loaderWorker=new Worker('lib/psd-worker.js');
      loaderWorker.onmessage=ev=>{
        if(ticket!==loadTicket)return;
        if(ev.data.progress){loadProgress(ev.data.progress);return;}
        loaderWorker.terminate();loaderWorker=null;cancelWorker=null;
        if(ev.data.error)reject(new Error(ev.data.error));else resolve(ev.data);
      };
      loaderWorker.onerror=ev=>{ev.preventDefault?.();loaderWorker?.terminate();loaderWorker=null;cancelWorker=null;reject(new Error('Could not start reading the PSD (possibly out of memory). Reopen the page from localhost or use a smaller PSD'));};
      loaderWorker.postMessage({buffer,generic},[buffer]);
    });
  }
  // Direct file opening cannot reliably start a worker. Keep that workflow usable.
  await new Promise(r=>setTimeout(r,20));
  if(ticket!==loadTicket)throw new DOMException('Cancelled','AbortError');
  const id=RT.fingerprint(buffer);
  const options={useImageData:true,skipThumbnail:true,skipCompositeImageData:true};
  Rigger.validatePsd(agPsd.readPsd(new Uint8Array(buffer),{...options,skipLayerImageData:true}));
  const psd=agPsd.readPsd(new Uint8Array(buffer),{...options,skipCompositeImageData:false}),pre=Rigger.cleanPsdLayers(psd);
  return {psd,pre,rig:Rigger.buildRig(psd,{generic}),id};
}
/** source(signal) resolves to an ArrayBuffer or {buffer, name}. */
async function loadModel(source,name,opts={}){
  cancelLoad();const ticket=loadTicket;
  loadController=new AbortController();const signal=loadController.signal;
  setLoading(true);
  dropEl.classList.remove('hidden');loadProgress(name+' — loading…');
  try{
    if(!window.agPsd)throw new Error('lib/ag-psd.min.js did not load. Check that the lib folder is next to the page');
    if(!window.Rigger)throw new Error('lib/rigger.js did not load. Check the lib folder');
    const got=await source(signal);
    let buf=got,displayName=name;
    if(got&&!(got instanceof ArrayBuffer)&&got.buffer instanceof ArrayBuffer){buf=got.buffer;displayName=got.name||name;}
    RT.validateHeader(buf);
    await customGenericReady;
    if(ticket!==loadTicket)return;
    let relayCopy=null;
    if(!OBS_MODE&&opts.relay!==false&&sync.mayRelay)relayCopy=buf.slice(0);   // kept so OBS can be (re)synced later
    if(ticket!==loadTicket)return;
    const parsed=await parseModel(buf,genericOpts().generic,ticket);
    if(ticket!==loadTicket)return;
    setAnchorMode(false);
    applyRig(parsed.rig);rigSimpleBlink=false;setSlider('simpleBlink',0);
    modelId=parsed.id||'';modelName=displayName;kamyiiModel=!!opts.kamyii;lastPsd=parsed.psd;lastPre=parsed.pre;
    $('modelName').textContent=displayName;$('modelName').title=displayName;
    document.title=displayName+' — Ka-myii Live Preview';
    resetParams();restoreSettings(true);
    history.reset(historySnapshot());savedSettings=comparableSettings();updateHistoryButtons();updateDirty();
    if(parsed.pre.noisy>0)$('rigInfo').textContent+='\nCleaned layers: '+parsed.pre.noisy+'/'+parsed.pre.layers+'Layers';
    dropStatus.textContent='';
    status(displayName+' — ready');
    if(relayCopy)sync.publishModel(relayCopy,displayName,modelId);
    syncState(true);
    if(OBS_MODE)afterObsModel();
  }catch(err){
    if(ticket!==loadTicket||err.name==='AbortError')return;
    dropStatus.textContent='Error: '+err.message;status(dropStatus.textContent,true);
    dropEl.classList.toggle('hidden',!!layers.length);
  }finally{
    if(ticket===loadTicket){setLoading(false);loadController=null;}
  }
}
$('btnCancelLoad').addEventListener('click',()=>{cancelLoad();status('Loading cancelled');});
let customGeneric=null;
async function loadCustomGeneric(){
  if(typeof fetch==='undefined'||!window.agPsd||!window.Rigger||location.protocol==='file:')return;
  async function grab(url){
    try{
      const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),8000);
      let buffer;
      try{const r=await fetch(url,{signal:controller.signal});if(!r.ok)return null;buffer=await r.arrayBuffer();}finally{clearTimeout(timer);}
      RT.validateHeader(buffer);
      const psd=window.agPsd.readPsd(new Uint8Array(buffer),{useImageData:true,skipThumbnail:true,skipCompositeImageData:true});
      return window.Rigger.flattenPsdToImg(psd);
    }catch(err){return null;}
  }
  const [eye,mouth]=await Promise.all([grab('eye_close.psd'),grab('mouth_close.psd')]);
  const g={};
  if(eye){const s=window.Rigger.splitImgLR(eye);if(s){g.eyeL=s.l;g.eyeR=s.r;}}
  if(mouth)g.mouth=mouth;
  if(g.eyeL||g.mouth)customGeneric=g;
}
const customGenericReady=loadCustomGeneric();
function genericOpts(){
  const GP=window.GenericParts;
  const base=GP?{eyeL:GP.get('eyeL'),eyeR:GP.get('eyeR'),mouth:GP.get('mouth')}:{};
  const g=Object.assign({},base,customGeneric||{});
  return (g.eyeL||g.mouth)?{generic:g}:{};
}
$('btnFile').addEventListener('click',()=>$('fileInput').click());
$('btnOpen').addEventListener('click',()=>$('fileInput').click());
function loadFile(f){
  return loadModel(async()=>{
    if(!/\.psd$/i.test(f.name))throw new Error('Choose a .psd file');
    if(f.size>RT.MAX_FILE_BYTES)throw new Error('The PSD must be 128 MB or smaller');
    return f.arrayBuffer();
  },f.name);
}
function openFile(f){if(/\.json$/i.test(f.name))importSettingsFile(f);else loadFile(f);}
$('fileInput').addEventListener('change',ev=>{const f=ev.target.files[0];ev.target.value='';if(f)openFile(f);});
function fetchSource(path){
  return async signal=>{
    const r=await fetch(path,{signal});
    if(!r.ok)throw new Error(path+' was not found (HTTP '+r.status+')');
    if(Number(r.headers.get('Content-Length'))>RT.MAX_FILE_BYTES)throw new Error('The PSD must be 128 MB or smaller');
    return r.arrayBuffer();
  };
}
const SAMPLE_FILES={A:'sample2.psd',B:'sample.psd'};
function loadSample(key='B'){const filename=SAMPLE_FILES[key];if(filename)return loadModel(fetchSource(filename),filename);}
document.querySelectorAll('[data-sample]').forEach(button=>button.addEventListener('click',()=>loadSample(button.dataset.sample)));
(function(){const miss=[];
  if(!window.agPsd)miss.push('ag-psd.min.js');if(!window.Rigger)miss.push('rigger.js');if(!window.GenericParts)miss.push('genericparts.js');
  if(miss.length)dropStatus.textContent='⚠ lib/'+miss.join(', lib/')+' did not load (check the lib folder)';
})();
// Drag & drop: PSD opens a model, JSON imports settings.
let dragDepth=0;
const hasFiles=e=>!!e.dataTransfer&&[...e.dataTransfer.types].includes('Files');
window.addEventListener('dragenter',e=>{if(!hasFiles(e)||OBS_MODE)return;e.preventDefault();dragDepth++;$('stage').classList.add('dragover');});
window.addEventListener('dragover',e=>{if(!hasFiles(e))return;e.preventDefault();e.dataTransfer.dropEffect='copy';});
window.addEventListener('dragleave',e=>{if(!hasFiles(e))return;dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)$('stage').classList.remove('dragover');});
window.addEventListener('drop',e=>{
  e.preventDefault();dragDepth=0;$('stage').classList.remove('dragover');
  const f=hasFiles(e)&&e.dataTransfer.files[0];if(f&&!OBS_MODE)openFile(f);
});

// ---------- parameters ----------
const P={angleX:0,angleY:0,angleZ:0,eyeOpenL:1,eyeOpenR:1,eyeX:0,eyeY:0,brow:0,
  mouthOpen:0,mouthForm:0,mouthTeeth:0,jawDrop:0,mouthExpressive:1,mouthCY:0,body:0,physAmp:0.95,soft:0.85,
  browAngL:0,browAngR:0,browAngSym:0,bangL:0,bangC:0,bangR:0,
  armY:0,armPos:0,shrug:0,bodyMotion:1,bodyTurn:0,weightShift:.4,breathing:1,bust:0.4,bustY:1,irisScale:1,mouthEase:0.72,eyeEase:0.3,mouthRound:0,mouthResponse:0.65,motionBounce:1,nodDepth:0,nodSpeed:2.6,swayDepth:0,
  fhAmp:0.95,fhSoft:0.65,eyeCY:0,eyeCAng:0,mouthCAng:0,eyeScaleL:1,eyeScaleR:1,mouthScale:1,accAmp:1,shadowFollow:1,eyeTiltL:0,eyeTiltR:0,simpleBlink:0,neko:0,tail:0,furPalette:0,earSize:1,earSpread:1,earLift:0,earMotion:1,tailSize:1,tailX:0,tailY:0,tailSide:1,tailMotion:1};
const DEFAULTS=Object.assign({},P);
const T=Object.assign({},P);
const cur=Object.assign({},P);
const auto={idle:true,blink:true,rand:false,talk:false,mouse:false,mic:false,phys:true,cam:false};
const autoDefaults={...auto};
if(OBS_MODE)$('tgTalk').classList.remove('on');

const sliders={pAngleX:'angleX',pAngleY:'angleY',pAngleZ:'angleZ',pEyeL:'eyeOpenL',pEyeR:'eyeOpenR',
  pEyeX:'eyeX',pEyeY:'eyeY',pBrow:'brow',pMouthOpen:'mouthOpen',pMouthForm:'mouthForm',pMouthTeeth:'mouthTeeth',pJawDrop:'jawDrop',
  pBody:'body',pPhysAmp:'physAmp',pSoft:'soft',pMouthCY:'mouthCY',
  pBrowAngL:'browAngL',pBrowAngR:'browAngR',pBrowAngSym:'browAngSym',
  pBangL:'bangL',pBangC:'bangC',pBangR:'bangR',pArmY:'armY',pArmPos:'armPos',pShrug:'shrug',pBodyMotion:'bodyMotion',pBodyTurn:'bodyTurn',pWeightShift:'weightShift',pBreathing:'breathing',
  pBust:'bust',pBustY:'bustY',pIrisScale:'irisScale',pMouthEase:'mouthEase',pEyeEase:'eyeEase',
  pFhAmp:'fhAmp',pFhSoft:'fhSoft',pEyeCY:'eyeCY',pEyeCAng:'eyeCAng',pMouthCAng:'mouthCAng',
  pEyeScaleL:'eyeScaleL',pEyeScaleR:'eyeScaleR',pMouthScale:'mouthScale',pAccAmp:'accAmp',pShadowFollow:'shadowFollow',pEyeTiltL:'eyeTiltL',pEyeTiltR:'eyeTiltR',pMouthRound:'mouthRound',pMouthResponse:'mouthResponse',pMotionBounce:'motionBounce',pNodDepth:'nodDepth',pNodSpeed:'nodSpeed',pSwayDepth:'swayDepth',
  pEarSize:'earSize',pEarSpread:'earSpread',pEarLift:'earLift',pEarMotion:'earMotion',pTailSize:'tailSize',pTailX:'tailX',pTailY:'tailY',pTailMotion:'tailMotion'};
const sliderOf={};
const parameterRanges={};
// Attach a number box to every range row. onValue(value, final) is called on edits.
function bindRange(el,onValue){
  const v=el.parentNode.querySelector('.val'),label=el.parentNode.querySelector('label');label.htmlFor=el.id;
  const number=document.createElement('input');number.type='number';number.className='val';number.min=el.min;number.max=el.max;number.step=el.step;number.value=el.value;
  number.setAttribute('aria-label',label.textContent+' value');v.replaceWith(number);
  const range=[Number(el.min),Number(el.max)];
  el.addEventListener('input',()=>{number.value=el.value;onValue(Number(el.value),false);});
  el.addEventListener('change',()=>onValue(Number(el.value),true));
  number.addEventListener('input',()=>{if(number.value!==''&&Number.isFinite(number.valueAsNumber)){el.value=RT.clamp(number.valueAsNumber,...range);onValue(Number(el.value),false);}});
  number.addEventListener('change',()=>{if(number.value!==''&&Number.isFinite(number.valueAsNumber))el.value=RT.clamp(number.valueAsNumber,...range);number.value=el.value;onValue(Number(el.value),true);});
  return {number,range,set(value){el.value=RT.clamp(value,...range);number.value=el.value;return Number(el.value);}};
}
for(const id in sliders){
  const el=$(id),key=sliders[id];
  const b=bindRange(el,(value,final)=>{
    T[key]=value;if(paused&&lastFrame)lastFrame[key]=value;
    if(activePreset){activePreset=null;highlightPresets();}
    invalidate();if(key==='bustY')renderOverlay();
    if(final)commitHistory();else syncState();
  });
  sliderOf[key]=b;parameterRanges[key]=b.range;
  el.addEventListener('dblclick',()=>{T[key]=b.set(DEFAULTS[key]);if(paused&&lastFrame)lastFrame[key]=T[key];invalidate();commitHistory();});
  el.title='Double-click to reset';T[key]=Number(el.value);
}
parameterRanges.simpleBlink=[0,1];
sliderOf.simpleBlink={range:[0,1],set(value){const v=value>=.5?1:0;$('blinkStyle').value=String(v);return v;}};
for(const [id,key,range] of [['mouthArtwork','mouthExpressive',[0,1]],['nekoStyle','neko',[0,1]],['tailStyle','tail',[0,1]],['furPalette','furPalette',[0,3]],['tailSide','tailSide',[-1,1]]]){
  const el=$(id);parameterRanges[key]=range;
  sliderOf[key]={range,set(value){el.value=String(key==='tailSide'?(value<0?-1:1):Math.round(value));return Number(el.value);}};
  el.addEventListener('change',()=>{setSlider(key,Number(el.value));commitHistory();});
}
for(const button of document.querySelectorAll('[data-motion]'))button.addEventListener('click',()=>{
  const amount=Number(button.dataset.motion);
  for(const [key,value] of Object.entries({nodDepth:0,swayDepth:0,motionBounce:amount,fhAmp:.55+.4*amount,fhSoft:.35+.3*amount,physAmp:.55+.4*amount,soft:.5+.35*amount}))setSlider(key,value);
  setAuto('phys',true);invalidate();commitHistory();
});
$('btnNaturalSway').addEventListener('click',()=>{
  for(const [key,value] of Object.entries({nodDepth:0,swayDepth:0,motionBounce:.4,bodyMotion:.85,weightShift:.9,breathing:1,fhAmp:.75,physAmp:.8}))setSlider(key,value);
  setAuto('idle',true);setAuto('phys',true);invalidate();commitHistory();
});
$('btnSpringNod').addEventListener('click',()=>{
  for(const [key,value] of Object.entries({nodDepth:.85,nodSpeed:2.6,swayDepth:0,motionBounce:1,fhAmp:.95,fhSoft:.8,physAmp:1,soft:.95}))setSlider(key,value);
  setAuto('idle',true);setAuto('phys',true);invalidate();commitHistory();
});
$('btnHeadSway').addEventListener('click',()=>{
  for(const [key,value] of Object.entries({swayDepth:.75,nodDepth:0,nodSpeed:2,motionBounce:1,fhAmp:.95,fhSoft:.8,physAmp:1,soft:.95}))setSlider(key,value);
  setAuto('idle',true);setAuto('phys',true);invalidate();commitHistory();
});
function rebuildBlinkArtwork(simple){
  if(!lastPsd)return;
  applyRig(Rigger.buildRig(lastPsd,{...genericOpts(),simpleBlink:!!simple}));rigSimpleBlink=!!simple;
}
function setSlider(key,val){
  const b=sliderOf[key];if(!b)return;
  T[key]=b.set(val);if(paused&&lastFrame)lastFrame[key]=T[key];invalidate();
}

const toggleIds={idle:'tgIdle',blink:'tgBlink',rand:'tgRand',talk:'tgTalk',cam:'tgCam',mouse:'tgMouse',mic:'tgMic',phys:'tgPhys'};
const toggleLabels={cam:'Use camera',mic:'Use microphone'};
function syncToggle(key){const el=$(toggleIds[key]);el.classList.toggle('on',auto[key]);el.setAttribute('aria-pressed',String(auto[key]));}
function setAuto(key,value){
  const changed=auto[key]!==value;
  auto[key]=value;syncToggle(key);
  if(key==='mic'){if(value)startMic();else mic.stop();}
  if(key==='cam'){if(value)startCam();else stopCam();}
  if(key==='mic'||key==='cam')updateLiveTicker();
  if(key==='blink'&&!value){blinkT=-1;blinkVariant=1;irisBounceT=-1;}
  if(changed&&RT.AUTO_KEYS.includes(key)){updateDirty();syncState();}
}
Object.entries(toggleIds).forEach(([key,id])=>{syncToggle(key);$(id).addEventListener('click',()=>setAuto(key,!auto[key]));});

const PRESET_ORDER=['neutral','smile','usume','surprise','jito','winkL','winkR'];
const presets={
  neutral:{eyeOpenL:1,eyeOpenR:1,brow:0,mouthOpen:0,mouthForm:0,mouthRound:0,mouthTeeth:0,jawDrop:0,irisScale:1},
  smile:{eyeOpenL:0,eyeOpenR:0,brow:0.45,mouthOpen:0,mouthForm:0.9,irisScale:1},
  usume:{eyeOpenL:0.5,eyeOpenR:0.5,brow:0.35,mouthOpen:1,mouthForm:0.8,irisScale:1},
  surprise:{eyeOpenL:1,eyeOpenR:1,brow:1,mouthOpen:.85,mouthForm:-.1,mouthRound:1,jawDrop:.35,irisScale:.7},
  jito:{eyeOpenL:0.4,eyeOpenR:0.4,brow:-0.6,mouthOpen:0,mouthForm:-0.4,irisScale:1},
  winkL:{eyeOpenL:0,eyeOpenR:1,brow:0.2,mouthOpen:0.4,mouthForm:0.7,irisScale:1},
  winkR:{eyeOpenL:1,eyeOpenR:0,brow:0.2,mouthOpen:0.4,mouthForm:0.7,irisScale:1},
  grin:{eyeOpenL:.85,eyeOpenR:.85,brow:.25,mouthOpen:.6,mouthForm:.9,mouthTeeth:1,irisScale:1},
  jawdrop:{eyeOpenL:1,eyeOpenR:1,brow:1,mouthOpen:1,mouthForm:-.25,mouthRound:1,jawDrop:1,irisScale:.75}};
let activePreset=null;
function highlightPresets(){document.querySelectorAll('[data-preset]').forEach(b=>{const on=b.dataset.preset===activePreset;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));});}
function togglePreset(name){
  if(!presets[name])return;
  if(activePreset===name){activePreset=null;for(const k in presets.neutral)setSlider(k,presets.neutral[k]);}
  else{for(const [k,v] of Object.entries({...presets.neutral,...presets[name]}))setSlider(k,v);activePreset=name;}
  highlightPresets();commitHistory();syncState(true);
}
function clearPreset(){if(!activePreset)return;togglePreset(activePreset);}
document.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',()=>togglePreset(b.dataset.preset)));
function resetParams(){
  activePreset=null;highlightPresets();
  for(const key in sliderOf)if(key!=='simpleBlink')setSlider(key,DEFAULTS[key]);
  Object.assign(cur,T);
}
$('btnReset').addEventListener('click',()=>{resetParams();commitHistory();$('utilStatus').textContent='All sliders reset';});
$('btnSavePsd').addEventListener('click',()=>{
  const st=$('utilStatus');
  if(!lastPsd){st.textContent='Load a PSD first';return;}
  try{
    st.textContent='Exporting…';
    const out=window.agPsd.writePsd(lastPsd,{generateThumbnail:false});
    const blob=new Blob([out],{type:'application/octet-stream'});
    download(blob,baseFileName()+'_clean.psd');
    st.textContent='Cleaned PSD saved ('+(blob.size/1e6).toFixed(1)+' MB, noise removed and trimmed)';
  }catch(err){st.textContent='ExportError: '+err.message;}
});
function setBackground(value){
  background=RT.BACKGROUNDS.includes(value)?value:'checker';
  const s=$('stage');s.classList.remove('checker','green','dark');s.classList.add(background);
  document.querySelectorAll('[data-bg]').forEach(b=>{b.classList.toggle('active',b.dataset.bg===background);b.setAttribute('aria-pressed',String(b.dataset.bg===background));});
}
document.querySelectorAll('[data-bg]').forEach(b=>b.addEventListener('click',()=>{setBackground(b.dataset.bg);commitHistory();}));
setBackground(background);

// ---------- settings, history, dirty state ----------
function settingsSnapshot(){
  return {format:RT.SETTINGS_FORMAT,version:RT.SETTINGS_VERSION,rigRevision:3,app:'Anime2.5DRig '+APP_VERSION,modelId,modelName,params:{...T},preset:activePreset,
    auto:Object.fromEntries(RT.AUTO_KEYS.map(k=>[k,auto[k]])),background,
    layers:layers.map(L=>({id:L.id,visible:L.visible,opacity:L.opacity,depth:L.depth})),anchors:JSON.parse(JSON.stringify(anchorOffsets))};
}
function comparableSettings(){const s=settingsSnapshot();delete s.modelName;delete s.app;return JSON.stringify(s);}
function defaultLayerOrder(){return layers.slice().sort((a,b)=>a.z-b.z).map(L=>L.id);}
function applySettings(value,opts={}){
  // One older both-eyes toggle becomes two blink-aware halves; keep visibility/opacity/depth.
  if(Array.isArray(value.layers))value={...value,layers:value.layers.flatMap(r=>{
    const children=layers.filter(L=>L.toggleKey&&r.id===Math.floor(L.z)+':'+L.toggleKey);
    return children.length?children.map(L=>({...r,id:L.id})):r;
  })};
  const data=RT.settings(value,modelId,parameterRanges,layers.map(L=>L.id),DEFAULTS,{anyModel:!!opts.anyModel,anchorLimit:Math.max(CW,CH)});
  if(!!data.params.simpleBlink!==rigSimpleBlink)rebuildBlinkArtwork(data.params.simpleBlink);
  data.missingLayers-=layers.filter(L=>(L.companion||L.expression)&&!value.layers.some(r=>r.id===L.id)).length;
  // Upgrade old automatic depth values; retain deliberate per-layer depth edits.
  if(!value.rigRevision){
    const legacy={facedetail:1.02,nose:1.15,eyewhite:1.06,irides:1.08,eyelash:1.12,eye_close:1.12,eye_close2:1.12,eyebrow:1.14,mouth_open:1.08,mouth_close:1.08};
    for(const r of data.layers){const L=layers.find(L=>L.id===r.id);if(L&&Math.abs(r.depth-legacy[L.bn])<1e-6)r.depth=L.defaultDepth;}
  }
  activePreset=null;
  for(const [key,n] of Object.entries(data.params))setSlider(key,n);
  activePreset=data.preset;highlightPresets();
  Object.assign(cur,T);
  if(opts.auto!==false)for(const [key,n] of Object.entries(data.auto))setAuto(key,n);
  const byId=new Map(layers.map(L=>[L.id,L])),records=new Map(data.layers.map(r=>[r.id,r]));
  layers=RT.mergeLayerOrder(defaultLayerOrder(),data.layers.map(r=>r.id)).map(id=>{
    const L=byId.get(id),r=records.get(id);
    if(r){L.visible=r.visible;L.opacity=r.opacity;L.depth=r.depth;}
    else{L.visible=L.fade!=='toggle';L.opacity=L.defaultOpacity;L.depth=L.defaultDepth;}
    return L;
  });
  anchorOffsets=data.anchors;applyAnchors();
  setBackground(data.background);renderLayerList();invalidate();
  return data;
}
function layerDiffNote(data){const n=data.missingLayers+data.unknownLayers;return n?' ('+n+' layer differences kept their defaults)':'';}
function restoreSettings(quiet=false){
  try{
    if(quiet)for(const k of RT.AUTO_KEYS)setAuto(k,autoDefaults[k]);
    const saved=storageGet('anime25d.settings.'+modelId);
    // Ka-myii keeps a character's adjustments in its folder: they survive preview rebuilds and project export.
    const kept=kamyiiModel&&KAMYII.settings;
    if(!saved&&!kept){if(!quiet)status('No saved adjustments for this character');return false;}
    const data=applySettings(kept||JSON.parse(saved),{anyModel:!!kept});
    $('utilStatus').textContent='Saved adjustments restored'+layerDiffNote(data);
    if(!quiet){commitHistory();savedSettings=comparableSettings();updateDirty();syncState(true);}
    return true;
  }catch(err){$('utilStatus').textContent='Could not restore adjustments: '+err.message;return false;}
}
function saveSettings(){
  if(!layers.length)return;
  const snapshot=settingsSnapshot();
  if(kamyiiModel){
    fetch('/api/generation/studio/'+encodeURIComponent(KAMYII.id),{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(snapshot)})
      .then(r=>r.json().then(d=>{if(!r.ok||!d.success)throw new Error(d.error||r.status);}))
      .then(()=>{KAMYII.settings=snapshot;savedSettings=comparableSettings();updateDirty();status('Adjustments saved with the character');})
      .catch(err=>status('Could not save the adjustments: '+err.message+'. Export the settings JSON instead',true));
    return;
  }
  try{storageSet('anime25d.settings.'+modelId,JSON.stringify(snapshot));savedSettings=comparableSettings();updateDirty();status('Adjustments saved in this browser');}
  catch(err){status('Could not save in this browser. Export the settings JSON instead',true);}
}
$('btnSaveSettings').addEventListener('click',saveSettings);
$('btnRestoreSettings').addEventListener('click',()=>restoreSettings());
$('btnExportSettings').addEventListener('click',()=>{download(new Blob([JSON.stringify(settingsSnapshot(),null,2)],{type:'application/json'}),baseFileName()+'.rig.json');status('Settings JSON exported');});
$('btnImportSettings').addEventListener('click',()=>$('settingsInput').click());
async function importSettingsFile(f){
  const id=modelId;
  try{
    if(!layers.length)throw new Error('Load a PSD first');
    if(f.size>1024*1024)throw new Error('The settings JSON must be 1 MB or smaller');
    const value=JSON.parse(await f.text());
    if(id!==modelId)throw new Error('The character changed. Choose the settings file again');
    const data=applySettings(value);commitHistory();syncState(true);
    status('Settings loaded'+layerDiffNote(data)+'. Use Save adjustments to keep them');
  }catch(err){status('Could not load the settings: '+err.message,true);}
}
$('settingsInput').addEventListener('change',ev=>{const f=ev.target.files[0];ev.target.value='';if(f)importSettingsFile(f);});

const history=RT.createHistory(150);
let savedSettings='',unsaved=false;
function historySnapshot(){
  const p={};for(const k in T)p[k]=round2(T[k]);
  return JSON.stringify({p,preset:activePreset,bg:background,a:anchorOffsets,l:layers.map(L=>[L.id,L.visible,round2(L.opacity),round2(L.depth)])});
}
function commitHistory(){
  if(!layers.length)return;
  history.push(historySnapshot());updateHistoryButtons();updateDirty();syncState();
}
function restoreHistory(snap,label){
  if(!snap)return;
  const s=JSON.parse(snap);
  if(!!s.p.simpleBlink!==rigSimpleBlink)rebuildBlinkArtwork(s.p.simpleBlink);
  applySettings({...settingsSnapshot(),params:s.p,preset:s.preset,background:s.bg,anchors:s.a||{},
    layers:s.l.map(([id,visible,opacity,depth])=>({id,visible,opacity,depth}))},{anyModel:true});
  updateHistoryButtons();updateDirty();syncState(true);
  status(label);
}
function undo(){if(history.canUndo)restoreHistory(history.undo(),'Undone');}
function redo(){if(history.canRedo)restoreHistory(history.redo(),'Redone');}
function updateHistoryButtons(){$('btnUndo').disabled=!history.canUndo;$('btnRedo').disabled=!history.canRedo;}
$('btnUndo').addEventListener('click',undo);$('btnRedo').addEventListener('click',redo);
function updateDirty(){
  unsaved=!!layers.length&&!OBS_MODE&&comparableSettings()!==savedSettings;
  const b=$('btnSaveSettings');b.textContent=unsaved?'Save adjustments ●':'Save adjustments';
  b.title=unsaved?'Unsaved changes (Ctrl+S)':(kamyiiModel?'Save with the character':'Save in this browser')+' (Ctrl+S)';
}
window.addEventListener('beforeunload',e=>{if(unsaved){e.preventDefault();e.returnValue='';}});

function togglePause(){
  paused=!paused;const b=$('btnPause');b.textContent=paused?'Play':'Pause';b.setAttribute('aria-pressed',String(paused));
  invalidate();status(paused?'Paused: adjust the sliders and save a still':'Playing');
}
$('btnPause').addEventListener('click',togglePause);
$('btnResetLayers').addEventListener('click',()=>{
  layers.sort((a,b)=>a.z-b.z);for(const L of layers){L.visible=L.fade!=='toggle';L.depth=L.defaultDepth;L.opacity=L.defaultOpacity;}
  renderLayerList();invalidate();commitHistory();status('Layers reset');
});

// ---------- export: PNG / video ----------
const recorder=window.RigRecorder?window.RigRecorder.create(cv):null;
function exportBackground(){return (capturePending||(recorder&&recorder.active))?EXPORT_BG[prefs.exportBg]||EXPORT_BG.transparent:EXPORT_BG.transparent;}
function capturePng(){if(layers.length){capturePending=true;invalidate();}}
$('btnPng').addEventListener('click',capturePng);$('btnPng2').addEventListener('click',capturePng);
function updateRecordAvailability(){
  const formats=window.RigRecorder?window.RigRecorder.formats():[];
  const ok=formats.length&&typeof cv.captureStream==='function';
  for(const id of ['btnRecord','btnRecord2'])if(!ok)$(id).disabled=true;
  if(!ok)$('recordStatus').textContent='This browser cannot record video (use Chrome, Edge or Firefox)';
}
function setRecordingUi(on){
  $('btnRecord').textContent=on?'Stop recording':'Record';$('btnRecord2').textContent=on?'Stop and save recording':'Start recording';
  $('btnRecord').classList.toggle('recording',on);$('btnRecord2').classList.toggle('recording',on);$('recBadge').hidden=!on;
}
async function toggleRecording(){
  if(!recorder||!layers.length)return;
  if(recorder.active){recorder.stop();return;}
  const fmt=prefs.recFormat;
  setRecordingUi(true);invalidate();
  try{
    const res=await recorder.start({seconds:prefs.recSeconds,fps:prefs.recFps,format:fmt,
      onProgress:(t,total)=>{$('recBadge').textContent='● REC '+t.toFixed(1)+' / '+total+'s';}});
    download(res.blob,baseFileName()+'.'+res.format.ext);
    const note=prefs.exportBg==='transparent'&&!res.format.alpha?' (this format does not keep transparency)':'';
    status('Video exported ('+cv.width+' × '+cv.height+' px, '+res.seconds.toFixed(1)+' s, '+(res.blob.size/1e6).toFixed(1)+' MB)'+note);
  }catch(err){
    if(err.name==='AbortError')status('Recording cancelled');else status('Could not record: '+err.message,true);
  }finally{setRecordingUi(false);invalidate();}
}
$('btnRecord').addEventListener('click',toggleRecording);$('btnRecord2').addEventListener('click',toggleRecording);

// ---------- dialogs (README / shortcuts) ----------
function mdToHtml(md){
  const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  const inline=s=>esc(s)
    .replace(/\*\*(.+?)\*\*/g,'<b>$1</b>')
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g,(_,text,url)=>/^(https?:\/\/|#)/i.test(url)||/^[\w\-./]+\.md(#[\w-]*)?$/.test(url)?'<a href="'+url+'" target="_blank" rel="noopener">'+text+'</a>':text);
  const lines=md.split(/\r?\n/);
  let html='',inCode=false,list=null,inTable=false;
  const closeList=()=>{if(list){html+='</'+list+'>';list=null;}};
  for(const l of lines){
    if(l.startsWith('```')){closeList();html+=inCode?'</code></pre>':'<pre><code>';inCode=!inCode;continue;}
    if(inCode){html+=esc(l)+'\n';continue;}
    if(/^\|/.test(l.trim())){
      closeList();
      if(/^\|[\s\-|:]+\|?$/.test(l.trim()))continue;
      const cells=l.trim().split('|').slice(1,-1).map(c=>inline(c.trim()));
      if(!inTable){html+='<table><tr>'+cells.map(c=>'<th>'+c+'</th>').join('')+'</tr>';inTable=true;}
      else html+='<tr>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>';
      continue;
    }else if(inTable){html+='</table>';inTable=false;}
    const ul=/^\s*[-*] /.test(l),ol=/^\s*\d+\. /.test(l);
    if(ul||ol){const tag=ul?'ul':'ol';if(list!==tag){closeList();html+='<'+tag+'>';list=tag;}
      html+='<li>'+inline(l.replace(/^\s*([-*]|\d+\.) /,''))+'</li>';continue;}
    closeList();
    const h=l.match(/^(#{1,4}) (.*)/);
    if(h){html+='<h'+h[1].length+'>'+inline(h[2])+'</h'+h[1].length+'>';continue;}
    if(/^> /.test(l)){html+='<blockquote>'+inline(l.slice(2))+'</blockquote>';continue;}
    if(l.trim()!=='')html+='<p>'+inline(l)+'</p>';
  }
  if(inCode)html+='</code></pre>';closeList();if(inTable)html+='</table>';
  return html;
}
let dialogFocus=null;
function openDialog(id,focusId){
  dialogFocus=document.activeElement;$(id).classList.add('show');$(focusId).focus();
}
function closeDialog(){
  const open=document.querySelector('.overlay-dialog.show');if(!open)return;
  open.classList.remove('show');dialogFocus?.focus?.();
}
let readmeLoaded=false;
async function openReadme(){
  openDialog('readmeOverlay','readmeClose');
  if(readmeLoaded)return;
  const body=$('readmeBody');
  try{
    const r=await fetch('README.md');
    if(!r.ok)throw new Error('HTTP '+r.status);
    body.innerHTML=mdToHtml(await r.text());readmeLoaded=true;
  }catch(err){
    body.innerHTML='<p>Could not load README.md. <a href="https://github.com/852wa/Anime2.5DRig#readme" target="_blank" rel="noopener">Read it on GitHub</a></p>';
  }
}
$('btnReadme').addEventListener('click',openReadme);$('btnReadme2').addEventListener('click',openReadme);
$('readmeClose').addEventListener('click',closeDialog);$('shortcutClose').addEventListener('click',closeDialog);
document.querySelectorAll('.overlay-dialog').forEach(d=>d.addEventListener('click',e=>{if(e.target===d)closeDialog();}));
const SHORTCUTS=[
  [['Space'],'Pause / Play'],[['1','–','7'],'Expressions (same key clears)'],[['0'],'Clear expression'],[['Shift','1','–','9'],'Switch toggles (blush, tears…)'],
  [['Ctrl','Z'],'Undo'],[['Ctrl','Shift','Z'],'Redo (or Ctrl+Y)'],[['Ctrl','S'],'Save adjustments in this browser'],
  [['Ctrl','O'],'Open PSD'],[['E'],'Edit anchors mode'],[['F'],'Fit the preview'],
  [['B'],'Change background'],[['Wheel'],'Zoom the preview'],[['Drag'],'Pan the zoomed preview'],
  [['Double-click'],'Fit the preview'],[['?'],'Show or hide this list'],[['Esc'],'Close dialogs and Edit anchors']];
$('shortcutTable').innerHTML=SHORTCUTS.map(([keys,desc])=>'<tr><td>'+keys.map(k=>k==='–'?'–':'<kbd>'+k+'</kbd>').join(' ')+'</td><td>'+desc+'</td></tr>').join('');
$('btnShortcuts').addEventListener('click',()=>openDialog('shortcutOverlay','shortcutClose'));

// ---------- keyboard ----------
document.addEventListener('keydown',e=>{
  const dlg=document.querySelector('.overlay-dialog.show');
  if(dlg){
    if(e.key==='Escape'){e.preventDefault();closeDialog();}
    if(e.key==='Tab'){
      const items=[...dlg.querySelectorAll('button,a[href]')];
      if(e.shiftKey&&document.activeElement===items[0]){e.preventDefault();items[items.length-1].focus();}
      else if(!e.shiftKey&&document.activeElement===items[items.length-1]){e.preventDefault();items[0].focus();}
    }
    return;
  }
  if(OBS_MODE)return;
  const t=e.target,tag=t.tagName;
  const textEntry=(tag==='INPUT'&&!['range','checkbox','button'].includes(t.type))||tag==='TEXTAREA'||tag==='SELECT'||t.isContentEditable;
  const mod=e.ctrlKey||e.metaKey,k=e.key.toLowerCase();
  if(mod&&!e.altKey){
    if(k==='s'){e.preventDefault();if(!e.repeat)saveSettings();return;}
    if(k==='o'){e.preventDefault();if(!e.repeat)$('fileInput').click();return;}
    if(textEntry)return;   // native undo inside text boxes
    if(k==='z'&&!e.shiftKey){e.preventDefault();undo();}
    else if((k==='z'&&e.shiftKey)||k==='y'){e.preventDefault();redo();}
    return;
  }
  if(textEntry||e.altKey||e.ctrlKey||e.metaKey)return;
  if(e.repeat&&!e.key.startsWith('Arrow')){if(e.code==='Space'||/^[0-7ebf?]$/i.test(e.key))e.preventDefault();return;}
  if(t.closest&&t.closest('#overlay .handle')&&e.key.startsWith('Arrow'))return;   // arrow keys move anchors
  if(e.code==='Space'){
    if(layers.length&&!/BUTTON|INPUT/.test(tag)&&!t.closest('[role="button"]')){e.preventDefault();togglePause();}
    return;
  }
  if(e.shiftKey&&/^Digit[1-9]$/.test(e.code)){const L=toggleLayers()[e.code.slice(5)-1];if(L){e.preventDefault();flipToggle(L);}return;}
  if(/^[0-7]$/.test(e.key)&&layers.length){e.preventDefault();if(e.key==='0')clearPreset();else togglePreset(PRESET_ORDER[Number(e.key)-1]);return;}
  if(k==='e'&&layers.length){e.preventDefault();setAnchorMode(!anchorMode);return;}
  if(k==='f'){e.preventDefault();resetView();return;}
  if(k==='b'){e.preventDefault();const i=RT.BACKGROUNDS.indexOf(background);setBackground(RT.BACKGROUNDS[(i+1)%RT.BACKGROUNDS.length]);commitHistory();return;}
  if(e.key==='?'){e.preventDefault();openDialog('shortcutOverlay','shortcutClose');return;}
  if(e.key==='Escape'&&anchorMode){e.preventDefault();setAnchorMode(false);}
});

// ---------- layer list ----------
const expandedLayers=new Set();let dragLayerIndex=-1;
function layerRole(L){
  const parts=[L.unknown?'Follows '+L.group:L.bn];
  if(L.side)parts.push(L.side==='L'?'left':'right');
  if(L.strands&&L.strands.length)parts.push(L.strands.length+' strands');
  if(L.synthetic)parts.push('generated');
  return parts.join(' · ');
}

function moveLayer(from,to){
  if(from===to||from<0||to<0||from>=layers.length||to>=layers.length)return;
  const [L]=layers.splice(from,1);layers.splice(to,0,L);
  renderLayerList();invalidate();commitHistory();
}
function toggleLayers(){
  const seen=new Set();return layers.filter(L=>{if(L.fade!=='toggle'||seen.has(L.bn))return false;seen.add(L.bn);return true;}).sort((a,b)=>a.z-b.z);
}
function flipToggle(L){
  const on=!layers.some(p=>p.bn===L.bn&&p.visible);
  // Competing designs are exclusive: one set of emote eyes, one alternate per part.
  for(const p of layers){if(p.bn===L.bn)p.visible=on;else if(on&&(L.toggleEye&&p.toggleEye||L.variantPart&&p.variantPart===L.variantPart))p.visible=false;}
  renderLayerList();invalidate();commitHistory();syncState(true);
}
function renderToggles(){
  const box=$('toggleBtns'),list=toggleLayers();if(!box)return;
  box.replaceChildren(...list.map((L,i)=>{
    const b=document.createElement('button'),name=L.bn.slice(7);b.type='button';b.className='btn'+(L.visible?' active':'');
    b.setAttribute('aria-pressed',String(L.visible));b.textContent=name[0].toUpperCase()+name.slice(1);
    if(i<9){const k=document.createElement('kbd');k.textContent='⇧'+(i+1);b.append(k);}
    b.addEventListener('click',()=>flipToggle(L));return b;}));
  box.hidden=!list.length;
}
function renderLayerList(){
  renderToggles();
  const container=$('layers');container.replaceChildren();
  layers.forEach((L,i)=>{
    const card=document.createElement('div');card.className='layer-card';card.classList.toggle('is-hidden',!L.visible);card.dataset.index=i;
    const top=document.createElement('div');top.className='layer-top';
    const thumb=document.createElement('img');thumb.className='layer-thumb';thumb.alt='';thumb.draggable=true;thumb.title='Drag to reorder';
    if(L.thumb)thumb.src=L.thumb;
    thumb.addEventListener('dragstart',e=>{dragLayerIndex=i;card.classList.add('dragging');e.dataTransfer.effectAllowed='move';e.dataTransfer.setData('text/x-anime25d-layer',String(i));});
    thumb.addEventListener('dragend',()=>{dragLayerIndex=-1;card.classList.remove('dragging');container.querySelectorAll('.drag-over').forEach(c=>c.classList.remove('drag-over'));});
    card.addEventListener('dragover',e=>{if(dragLayerIndex<0)return;e.preventDefault();card.classList.add('drag-over');});
    card.addEventListener('dragleave',()=>card.classList.remove('drag-over'));
    card.addEventListener('drop',e=>{if(dragLayerIndex<0)return;e.preventDefault();e.stopPropagation();moveLayer(dragLayerIndex,i);});
    const visible=document.createElement('input');visible.type='checkbox';visible.checked=L.visible;visible.setAttribute('aria-label',(L.source||L.name)+' visibility');
    visible.addEventListener('change',()=>{L.visible=visible.checked;card.classList.toggle('is-hidden',!L.visible);invalidate();commitHistory();});
    const text=document.createElement('div');text.className='layer-text';
    const name=document.createElement('span');name.className='layer-name';name.textContent=L.source||L.name;name.title=(L.source||L.name)+(L.source&&L.source!==L.name?' → '+L.name:'');
    const role=document.createElement('span');role.className='layer-role';role.classList.toggle('unknown',!!L.unknown);role.textContent=layerRole(L);
    text.append(name,role);
    top.append(thumb,visible,text);
    for(const d of [-1,1]){
      const b=document.createElement('button');b.type='button';b.className='ord';b.textContent=d<0?'↑':'↓';
      b.setAttribute('aria-label',(L.source||L.name)+(d<0?' backward':' forward'));b.disabled=i+d<0||i+d>=layers.length;
      b.addEventListener('click',()=>{moveLayer(i,i+d);container.children[i+d]?.querySelector(d<0?'.ord':'.ord:nth-of-type(2)')?.focus();});
      top.append(b);
    }
    const more=document.createElement('button');more.type='button';more.className='more';
    const open=expandedLayers.has(L.id);more.textContent=open?'▾':'▸';more.setAttribute('aria-expanded',String(open));more.setAttribute('aria-label',(L.source||L.name)+' details');
    top.append(more);card.append(top);
    const details=document.createElement('div');details.className='layer-details';details.hidden=!open;
    more.addEventListener('click',()=>{const now=!expandedLayers.has(L.id);if(now)expandedLayers.add(L.id);else expandedLayers.delete(L.id);details.hidden=!now;more.textContent=now?'▾':'▸';more.setAttribute('aria-expanded',String(now));});
    for(const [key,label,max] of [['opacity','Opacity',1],['depth','Depth',2]]){
      const row=document.createElement('div');row.className='row';const caption=document.createElement('label');caption.textContent=label;
      const range=document.createElement('input');range.type='range';range.min='0';range.max=String(max);range.step='0.01';range.value=L[key];range.id='layer-'+i+'-'+key;caption.htmlFor=range.id;range.setAttribute('aria-label',(L.source||L.name)+' '+label);
      const value=document.createElement('input');value.type='number';value.min='0';value.max=String(max);value.step='0.01';value.value=L[key];value.setAttribute('aria-label',(L.source||L.name)+' '+label+' value');
      range.addEventListener('input',()=>{L[key]=Number(range.value);value.value=range.value;invalidate();syncState();});
      range.addEventListener('change',commitHistory);
      value.addEventListener('input',()=>{if(value.value!==''&&Number.isFinite(value.valueAsNumber)){L[key]=RT.clamp(value.valueAsNumber,0,max);range.value=L[key];invalidate();}});
      value.addEventListener('change',()=>{if(value.value!==''&&Number.isFinite(value.valueAsNumber))L[key]=RT.clamp(value.valueAsNumber,0,max);range.value=L[key];value.value=L[key];invalidate();commitHistory();});
      row.append(caption,range,value);details.append(row);
    }
    card.append(details);
    card.addEventListener('mouseenter',()=>{highlightLayer=L.id;renderOverlay();});
    card.addEventListener('mouseleave',()=>{if(highlightLayer===L.id){highlightLayer=null;renderOverlay();}});
    container.append(card);
  });
  if(!layers.length){const p=document.createElement('p');p.className='section-hint';p.textContent='Layers appear after loading';container.append(p);}
  applySearch();
}

// ---------- diagnostics ----------
function renderDiagnostics(rig){
  const list=$('diagnostics');list.replaceChildren();
  const add=(cls,text)=>{const li=document.createElement('li');li.className=cls;li.textContent=text;list.append(li);};
  for(const [bn,label] of [['face','Face'],['eyewhite','Eye whites'],['irides','Irises'],['eyelash','Eyelashes'],['eyebrow','Eyebrows'],['mouth_open','Mouth']]){
    const found=layers.filter(L=>L.bn===bn);
    add(found.length?'ok':'warn',label+(found.length?' detected':' missing — motion may be limited'));
  }
  add(rig.synth.eye?'auto':'ok',rig.synth.eye?'Closed eyes generated from matching tinted artwork':'Closed eyes supplied in artwork');
  add(rig.synth.mouth?'auto':'ok',rig.synth.mouth?'Closed mouth generated from tinted artwork':'Closed mouth supplied in artwork');
  const unknown=layers.filter(L=>L.unknown);
  if(unknown.length)add('warn','Unclassified layers: '+unknown.map(L=>L.source||L.name).join(', '));
  $('rigInfo').textContent=layers.length+' animated parts / '+CW+' × '+CH+' px';
}


// ---------- panel: collapse & search ----------
function collapseSection(sec,value){sec.classList.toggle('collapsed',value);sec.querySelector('h2').setAttribute('aria-expanded',String(!value));}
document.querySelectorAll('#panel .sec > h2').forEach(h=>{
  h.tabIndex=0;h.setAttribute('role','button');h.setAttribute('aria-expanded','true');
  h.addEventListener('click',()=>{collapseSection(h.parentNode,!h.parentNode.classList.contains('collapsed'));saveCollapsed();});
  h.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();h.click();}});
});
function saveCollapsed(){
  prefs.collapsed=[...document.querySelectorAll('#panel .sec.collapsed')].map(s=>s.dataset.sec);savePrefs();
  const all=[...document.querySelectorAll('#panel .sec')];$('btnCollapse').textContent=all.every(s=>s.classList.contains('collapsed'))?'Expand all':'Collapse all';
}
$('btnCollapse').addEventListener('click',()=>{const all=[...document.querySelectorAll('#panel .sec')],value=all.some(s=>!s.classList.contains('collapsed'));for(const s of all)collapseSection(s,value);saveCollapsed();});
const FILTER_ITEMS='.row,.toggles>*,.btns>*,.layer-card,.check-row,.field-row,.copy-row,.meter';
function applySearch(){
  const q=$('controlSearch').value.trim().toLowerCase();let matches=0;
  document.querySelectorAll('#panel .sec').forEach(sec=>{
    const items=[...sec.querySelectorAll(FILTER_ITEMS)];
    items.forEach(i=>i.classList.remove('filtered-out'));
    sec.querySelectorAll('.section-hint').forEach(i=>i.classList.remove('filtered-out'));
    if(!q){sec.hidden=false;matches++;return;}
    const title=sec.querySelector('h2').textContent.toLowerCase().includes(q);
    let any=title;
    if(!title){
      for(const i of items){const hit=(i.textContent+' '+(i.getAttribute('aria-label')||'')+' '+(i.title||'')+' '+[...i.querySelectorAll('[aria-label]')].map(x=>x.getAttribute('aria-label')).join(' ')).toLowerCase().includes(q);i.classList.toggle('filtered-out',!hit);if(hit)any=true;}
      sec.querySelectorAll('.section-hint').forEach(i=>i.classList.add('filtered-out'));
    }
    sec.hidden=!any;if(any){matches++;collapseSection(sec,false);}
  });
  $('searchEmpty').hidden=matches>0;
}
$('controlSearch').addEventListener('input',applySearch);

// ---------- preview: fit, zoom & pan ----------
const stage=$('stage'),vp=$('viewport'),view={z:1,x:0,y:0};let fitScale=1,baseLeft=0,baseTop=0;
function fit(){
  const W=stage.clientWidth,H=stage.clientHeight;
  fitScale=Math.max(1e-3,Math.min(W/renderBox.w,H/renderBox.h));
  const w=renderBox.w*fitScale,h=renderBox.h*fitScale;baseLeft=(W-w)/2;baseTop=(H-h)/2;
  vp.style.width=w+'px';vp.style.height=h+'px';applyView();
}
function applyView(){
  vp.style.transform='translate('+(baseLeft+view.x)+'px,'+(baseTop+view.y)+'px) scale('+view.z+')';
  const isFit=view.z===1&&!view.x&&!view.y;
  $('btnZoom').textContent=isFit?'Fit':Math.round(fitScale*view.z*100)+'%';
  renderOverlay();
}
function resetView(){view.z=1;view.x=0;view.y=0;applyView();}
function zoomAt(factor,cx,cy){
  const z2=clamp(view.z*factor,0.5,16);
  const lx=(cx-baseLeft-view.x)/view.z,ly=(cy-baseTop-view.y)/view.z;
  view.x=cx-baseLeft-z2*lx;view.y=cy-baseTop-z2*ly;view.z=z2;applyView();
}
window.addEventListener('resize',fit);
if(typeof ResizeObserver!=='undefined')new ResizeObserver(fit).observe(stage);
$('btnZoom').addEventListener('click',resetView);
if(KAMYII){
  const portraitButton=document.createElement('button');portraitButton.type='button';portraitButton.className='stage-btn';portraitButton.textContent='Face detail';portraitButton.id='btnPortrait';
  portraitButton.addEventListener('click',()=>{
    if(!A)return;
    const faceHeight=Math.max(1,A.face.y1-A.face.y0);
    view.z=clamp(CH/(faceHeight*3.8),1,8);
    view.x=stage.clientWidth/2-baseLeft-(A.face.cx-renderBox.left)*fitScale*view.z;
    view.y=stage.clientHeight*0.35-baseTop-(A.face.cy-renderBox.top)*fitScale*view.z;
    applyView();
  });
  $('stageTools').insertBefore(portraitButton,$('btnZoom'));
}

const pointers=new Map();let pinch=null;
function stagePoint(e){const r=stage.getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top};}
function panTarget(e){return !OBS_MODE&&layers.length&&!e.target.closest('#drop,#stageTools,#camPreview,#overlay .handle,button');}
if(!OBS_MODE){
  stage.addEventListener('wheel',e=>{
    if(!layers.length||e.target.closest('#drop,#camPreview'))return;
    e.preventDefault();const p=stagePoint(e),dy=e.deltaMode===1?e.deltaY*16:e.deltaY;
    zoomAt(Math.exp(-dy*0.0015),p.x,p.y);
  },{passive:false});
  stage.addEventListener('pointerdown',e=>{
    if(!panTarget(e)||(e.pointerType==='mouse'&&e.button!==0&&e.button!==1))return;
    stage.setPointerCapture(e.pointerId);pointers.set(e.pointerId,stagePoint(e));
    if(pointers.size===2){const [a,b]=[...pointers.values()];pinch={d:Math.hypot(a.x-b.x,a.y-b.y),m:{x:(a.x+b.x)/2,y:(a.y+b.y)/2}};}
    stage.classList.add('panning');
  });
  stage.addEventListener('pointermove',e=>{
    if(!pointers.has(e.pointerId))return;
    const prev=pointers.get(e.pointerId),p=stagePoint(e);pointers.set(e.pointerId,p);
    if(pointers.size===1){view.x+=p.x-prev.x;view.y+=p.y-prev.y;applyView();}
    else if(pointers.size===2&&pinch){
      const [a,b]=[...pointers.values()],d=Math.hypot(a.x-b.x,a.y-b.y),m={x:(a.x+b.x)/2,y:(a.y+b.y)/2};
      view.x+=m.x-pinch.m.x;view.y+=m.y-pinch.m.y;zoomAt(d/Math.max(1,pinch.d),m.x,m.y);pinch={d,m};
    }
  });
  const endPointer=e=>{pointers.delete(e.pointerId);if(pointers.size<2)pinch=null;if(!pointers.size)stage.classList.remove('panning');};
  stage.addEventListener('pointerup',endPointer);stage.addEventListener('pointercancel',endPointer);
  stage.addEventListener('dblclick',e=>{if(panTarget(e))resetView();});
}

// ---------- overlay: anchors & layer highlight ----------
const overlay=$('overlay'),SVG_NS='http://www.w3.org/2000/svg';
function svg(tag,attrs){const el=document.createElementNS(SVG_NS,tag);for(const k in attrs)el.setAttribute(k,attrs[k]);return el;}
function anchorHandles(){
  const out=[];if(!A)return out;
  out.push({key:'face',label:'Face centre',x:A.face.cx,y:A.face.cy,axes:['dx','dy']});
  for(const s of ['L','R']){
    const e=A['eye'+s];if(!e)continue;const name=s==='L'?'Left eye':'Right eye';
    out.push({key:'eye'+s,label:name,x:e.icx,y:e.icy,axes:['dx','dy'],place:'above'});
    out.push({key:'eye'+s+'Close',label:name+' closed position',x:blinkEditor?e.closeX:(s==='L'?e.x0-4:e.x1+4),y:e.closeY+(blinkEditor?T.eyeCY*14*FS:0),axes:['dx','dy'],cls:'close',guide:[e.x0,e.x1],place:s==='L'?'left':'right',eye:e});
  }
  out.push({key:'mouth',label:'Mouth',x:A.mouth.cx,y:A.mouth.cy,axes:['dx','dy']});
  out.push({key:'neck',label:'Neck pivot',x:NP.cx,y:NP.cy,axes:['dx','dy']});
  if(layers.some(L=>L.bn==='topwear'))out.push({key:'chest',label:'Chest',x:CHEST.cx,y:CHEST.cy+T.bustY*70*FS,axes:['dx','dy'],cls:'chest'});
  return blinkEditor?out.filter(h=>h.cls==='close'):out;
}
let dragAnchor=null;
function psdPerPx(){const r=overlay.getBoundingClientRect();return r.width>0?renderBox.w/r.width:1;}
function renderOverlay(){
  if(OBS_MODE)return;
  overlay.replaceChildren();
  if(!layers.length)return;
  const u=psdPerPx();
  // Sizes are given in PSD units per screen pixel so handles stay the same size at any zoom.
  if(highlightLayer){const L=layers.find(x=>x.id===highlightLayer);if(L)overlay.append(svg('rect',{class:'layer-box',x:L.x,y:L.y,width:L.w,height:L.h,'stroke-width':1.5*u}));}
  if(!anchorMode)return;
  for(const h of anchorHandles()){
    if(h.guide)overlay.append(svg('line',{class:'guide',x1:h.guide[0],x2:h.guide[1],y1:h.y,y2:h.y,'stroke-width':u,'stroke-dasharray':4*u+' '+4*u}));
    const g=svg('g',{class:'handle'+(h.cls?' '+h.cls:''),tabindex:'0',role:'button','aria-label':h.label+' (drag or use the arrow keys)','data-key':h.key});
    if(dragAnchor&&dragAnchor.key===h.key)g.classList.add('dragging');
    if(blinkEditor){
      const w=(h.eye.x1-h.eye.x0)*1.25,height=Math.max(18*FS,(h.eye.y1-h.eye.y0)*.8);
      g.append(svg('rect',{class:'blink-box',x:h.x-w/2,y:h.y-height/2,width:w,height,'stroke-width':u}));
      const original=baseAnchors[h.key.slice(0,4)];
      g.append(svg('line',{class:'blink-origin',x1:original.x0,x2:original.x1,y1:original.closeY,y2:original.closeY,'stroke-width':u,'stroke-dasharray':2*u+' '+3*u}));
    }
    g.append(svg('circle',{cx:h.x,cy:h.y,r:7*u,'stroke-width':2*u}));
    const place=h.place||'right';
    const tx=svg('text',{x:place==='left'?h.x-11*u:place==='above'?h.x:h.x+11*u,y:place==='above'?h.y-12*u:h.y+4*u,'font-size':12*u,'stroke-width':3*u,
      'text-anchor':place==='left'?'end':place==='above'?'middle':'start'});
    tx.textContent=h.label;g.append(tx);
    g.addEventListener('pointerdown',e=>startAnchorDrag(e,h));
    g.addEventListener('keydown',e=>nudgeAnchor(e,h));
    overlay.append(g);
  }
}
function toPsd(e){const r=overlay.getBoundingClientRect();return {x:(e.clientX-r.left)/r.width*renderBox.w+renderBox.left,y:(e.clientY-r.top)/r.height*renderBox.h+renderBox.top};}
function startAnchorDrag(e,h){
  if(e.button!==undefined&&e.button!==0)return;
  e.preventDefault();e.stopPropagation();
  const id=e.pointerId,start=toPsd(e),base=Object.assign({dx:0,dy:0},anchorOffsets[h.key]);
  dragAnchor={key:h.key};
  if(h.cls==='close')anchorPreview=h.key.slice(3,4);
  // Handles are re-created on every move, so follow the pointer on window.
  const move=ev=>{
    if(ev.pointerId!==id)return;
    const p=toPsd(ev),o={};
    if(h.axes.includes('dx'))o.dx=Math.round(base.dx+p.x-start.x);
    if(h.axes.includes('dy'))o.dy=Math.round(base.dy+p.y-start.y);
    setAnchorOffset(h.key,o);
  };
  const up=ev=>{
    if(ev.pointerId!==id)return;
    window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);window.removeEventListener('pointercancel',up);
    dragAnchor=null;if(!blinkEditor)anchorPreview=null;commitHistory();renderOverlay();invalidate();
    overlay.querySelector('[data-key="'+h.key+'"]')?.focus({preventScroll:true});
  };
  window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);window.addEventListener('pointercancel',up);
}
function setAnchorOffset(key,o){
  const clean=RT.anchors({[key]:o},Math.max(CW,CH));
  if(clean[key])anchorOffsets[key]=clean[key];else delete anchorOffsets[key];
  applyAnchors();syncState();
}
function nudgeAnchor(e,h){
  const step=e.shiftKey?10:1,d={ArrowLeft:[-step,0],ArrowRight:[step,0],ArrowUp:[0,-step],ArrowDown:[0,step]}[e.key];
  if(!d)return;
  e.preventDefault();e.stopPropagation();
  const o=Object.assign({dx:0,dy:0},anchorOffsets[h.key]);
  if(h.axes.includes('dx'))o.dx+=d[0];if(h.axes.includes('dy'))o.dy+=d[1];
  if(!h.axes.includes('dx'))delete o.dx;
  setAnchorOffset(h.key,o);commitHistory();
  overlay.querySelector('[data-key="'+h.key+'"]')?.focus();
}
function setAnchorMode(on){
  on=!!on&&layers.length>0&&!OBS_MODE;
  if(anchorMode===on)return;
  anchorMode=on;anchorPreview=null;
  if(!on)blinkEditor=false;
  syncBlinkEditor();
  for(const id of ['btnAnchorMode','btnAnchorMode2'])$(id).setAttribute('aria-pressed',String(on));
  $('anchorHint').hidden=!on;
  if(!on){Object.assign(cur,T);lastFrame=null;}
  renderOverlay();invalidate();
  if(on)status(blinkEditor?'Drag each closed eye; arrow keys nudge. Save adjustments when aligned.':'Editing anchors: adjust the reference points on a still pose');
}
$('btnAnchorMode').addEventListener('click',()=>setAnchorMode(!anchorMode));
$('btnAnchorMode2').addEventListener('click',()=>setAnchorMode(!anchorMode));
$('btnResetAnchors').addEventListener('click',()=>{anchorOffsets={};applyAnchors();commitHistory();syncState(true);status('Anchors reset to the detected positions');});

// ---------- manual blink placement (saved with existing character settings) ----------
function syncBlinkEditor(){
  for(const id of ['btnBlinkEdit','btnBlinkEdit2'])$(id).setAttribute('aria-pressed',String(blinkEditor));
  $('blinkSection').classList.toggle('editing',blinkEditor);
  for(const side of ['L','R'])for(const [axis,suffix] of [['dx','X'],['dy','Y']]){
    const el=$('blink'+side+suffix),value=anchorOffsets['eye'+side+'Close']?.[axis]||0;
    if(document.activeElement!==el)el.value=value;
    el.disabled=!A?.['eye'+side];
  }
  for(const button of document.querySelectorAll('[data-blink-view]'))button.setAttribute('aria-pressed',String(button.dataset.blinkView===blinkView));
  $('blinkHelp').textContent=blinkEditor?'Drag a box on the stage, or enter offsets. Arrow keys: 1 px; Shift: 10 px. Save adjustments keeps this on your character.':'Placement affects the preview. Use Test motion to check it, then Save adjustments. Your original artwork stays intact.';
}
function editBlink(on=true){
  if(!layers.length)return;
  const was=blinkEditor;blinkEditor=!!on;
  if(on){
    setAnchorMode(true);syncBlinkEditor();
    $('controlSearch').value='';applySearch();collapseSection($('blinkSection'),false);
    // Focus the drawing so small full-body eyes are easy to grab.
    if(!was){
      view.z=clamp(stage.clientWidth/(Math.max(1,A.face.x1-A.face.x0)*1.8*fitScale),1,10);
      view.x=stage.clientWidth/2-baseLeft-(A.face.cx-renderBox.left)*fitScale*view.z;
      view.y=stage.clientHeight*.48-baseTop-(A.face.cy-renderBox.top)*fitScale*view.z;applyView();
    }
    if(stage.getBoundingClientRect().right<$('panel').getBoundingClientRect().left+2)$('blinkSection').scrollIntoView({block:'nearest'});
  }else setAnchorMode(false);
  renderOverlay();invalidate();
}
for(const id of ['btnBlinkEdit','btnBlinkEdit2'])$(id).addEventListener('click',()=>editBlink(!blinkEditor));
$('btnBlinkDone').addEventListener('click',()=>{editBlink(false);if(paused)togglePause();});
for(const button of document.querySelectorAll('[data-blink-view]'))button.addEventListener('click',()=>{blinkView=button.dataset.blinkView;editBlink();});
for(const side of ['L','R'])for(const [axis,suffix] of [['dx','X'],['dy','Y']]){
  const el=$('blink'+side+suffix);
  el.addEventListener('input',()=>{
    if(!Number.isFinite(el.valueAsNumber))return;
    editBlink();const offset={...anchorOffsets['eye'+side+'Close'],[axis]:clamp(el.valueAsNumber,-200,200)};el.value=offset[axis];
    setAnchorOffset('eye'+side+'Close',offset);
  });
  el.addEventListener('change',()=>{syncBlinkEditor();commitHistory();});
}
$('blinkStyle').addEventListener('change',()=>{
  const snapshot=settingsSnapshot();snapshot.params.simpleBlink=Number($('blinkStyle').value);
  // Alignment for one drawing is not a reliable starting point for a different lid.
  delete snapshot.anchors.eyeLClose;delete snapshot.anchors.eyeRClose;
  for(const key of ['eyeTiltL','eyeTiltR','eyeScaleL','eyeScaleR','eyeCY','eyeCAng'])snapshot.params[key]=DEFAULTS[key];
  applySettings(snapshot,{anyModel:true});editBlink();commitHistory();
  status(rigSimpleBlink?'Clean fallback blink ready to position; original painting is preserved':'Original painted blink restored');
});
$('btnBlinkMatch').addEventListener('click',()=>{
  editBlink();
  for(const side of ['L','R']){
    const eye=baseAnchors['eye'+side];if(!eye)continue;
    const dy=Math.round(eye.y0+(eye.y1-eye.y0)*.62-eye.closeY-T.eyeCY*14*FS);
    setAnchorOffset('eye'+side+'Close',{...anchorOffsets['eye'+side+'Close'],dy});
  }
  commitHistory();status('Closed lids placed on each open eye’s natural closing line; refine with the boxes');
});
$('btnBlinkReset').addEventListener('click',()=>{
  delete anchorOffsets.eyeLClose;delete anchorOffsets.eyeRClose;
  for(const key of ['eyeTiltL','eyeTiltR','eyeScaleL','eyeScaleR','eyeCY','eyeCAng'])setSlider(key,DEFAULTS[key]);
  applyAnchors();commitHistory();status('Blink alignment reset');
});
$('btnBlinkSave').addEventListener('click',saveSettings);

// ---------- preferences (per browser) ----------
const PREF_SCHEMA={
  headGain:[0.3,2,1],eyeGain:[0.5,2,1],mouthGain:[0.5,2.5,1],browGain:[0,2,1],gazeGain:[0,2,1],smoothing:[0,1,0.5],
  linkEyes:['bool',true],trackBrow:['bool',true],trackSmile:['bool',true],camPreview:['bool',true],
  micGain:[0.25,4,1],micGate:[0,0.5,0.05],
  exportBg:['enum',['transparent','green','dark'],'transparent'],recSeconds:[1,60,5],recFps:['enum',[30,60],30],recFormat:['any',''],
  obsSync:['bool',true],collapsed:['any',[]],calibration:['any',null]};
let prefs=RT.prefs(null,PREF_SCHEMA);
try{prefs=RT.prefs(JSON.parse(storageGet('anime25d.prefs')||'null'),PREF_SCHEMA);}catch(err){}
if(!prefs.calibration||typeof prefs.calibration!=='object'||!Object.keys(FF.DEFAULT_NEUTRAL).every(k=>Number.isFinite(prefs.calibration[k])))prefs.calibration=null;
if(!Array.isArray(prefs.collapsed))prefs.collapsed=[];
let prefTimer=null;
function savePrefs(){clearTimeout(prefTimer);prefTimer=setTimeout(()=>{try{storageSet('anime25d.prefs',JSON.stringify(prefs));}catch(err){}},200);}
const prefSliders={prefHeadGain:'headGain',prefEyeGain:'eyeGain',prefMouthGain:'mouthGain',prefBrowGain:'browGain',prefGazeGain:'gazeGain',prefSmoothing:'smoothing',prefMicGain:'micGain',prefMicGate:'micGate'};
for(const [id,key] of Object.entries(prefSliders)){
  const el=$(id);el.value=prefs[key];
  const b=bindRange(el,value=>{prefs[key]=value;savePrefs();tracker.setOptions(trackerOptions());});
  el.addEventListener('dblclick',()=>{prefs[key]=b.set(PREF_SCHEMA[key][2]);savePrefs();tracker.setOptions(trackerOptions());});
  el.title='Double-click to reset';
}
const prefChecks={prefLinkEyes:'linkEyes',prefTrackBrow:'trackBrow',prefTrackSmile:'trackSmile',prefCamPreview:'camPreview',prefObsSync:'obsSync'};
for(const [id,key] of Object.entries(prefChecks)){
  const el=$(id);el.checked=prefs[key];
  el.addEventListener('change',()=>{prefs[key]=el.checked;savePrefs();tracker.setOptions(trackerOptions());if(key==='camPreview')updateCamPreview();if(key==='obsSync'){sync.setEnabled(el.checked);syncState(true);}});
}
const formatSelect=$('prefRecFormat');
(window.RigRecorder?window.RigRecorder.formats():[]).forEach(f=>{const o=document.createElement('option');o.value=f.mime;o.textContent=f.label;formatSelect.append(o);});
if(!formatSelect.options.length){const o=document.createElement('option');o.textContent='(not supported)';formatSelect.append(o);formatSelect.disabled=true;}
if([...formatSelect.options].some(o=>o.value===prefs.recFormat))formatSelect.value=prefs.recFormat;else prefs.recFormat=formatSelect.value;
$('prefExportBg').value=prefs.exportBg;$('prefRecSeconds').value=String(prefs.recSeconds);$('prefRecFps').value=String(prefs.recFps);
if(!$('prefRecSeconds').value)$('prefRecSeconds').value='5';
$('prefExportBg').addEventListener('change',e=>{prefs.exportBg=e.target.value;savePrefs();});
$('prefRecSeconds').addEventListener('change',e=>{prefs.recSeconds=Number(e.target.value);savePrefs();});
$('prefRecFps').addEventListener('change',e=>{prefs.recFps=Number(e.target.value);savePrefs();});
formatSelect.addEventListener('change',e=>{prefs.recFormat=e.target.value;savePrefs();});
for(const key of prefs.collapsed){const s=document.querySelector('#panel .sec[data-sec="'+key+'"]');if(s)collapseSection(s,true);}

// ---------- mouse follow ----------
const mouse={x:0,y:0,in:false};
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();mouse.x=clamp((e.clientX-r.left)/r.width*2-1,-1.5,1.5);mouse.y=clamp((e.clientY-r.top)/r.height*2-1,-1.5,1.5);mouse.in=true;});
cv.addEventListener('mouseleave',()=>mouse.in=false);

// ---------- camera & microphone ----------
function trackerOptions(){return {headGain:prefs.headGain,eyeGain:prefs.eyeGain,mouthGain:prefs.mouthGain,browGain:prefs.browGain,gazeGain:prefs.gazeGain,smoothing:prefs.smoothing,linkEyes:prefs.linkEyes,trackBrow:prefs.trackBrow,trackSmile:prefs.trackSmile};}
const tracker=new FF.Tracker(trackerOptions());
tracker.setCalibration(prefs.calibration);
const cam={live:false,ax:0,ay:0,az:0,eL:1,eR:1,mo:0,ex:0,ey:0,br:0,mf:0};
let lastTrackingAt=0,camPhysScale=1,remoteMic=0,remoteMicAt=0,micLevel=0;
function camChip(text,busy){const el=$('tgCam');el.textContent=text;el.classList.toggle('busy',!!busy);}
const camera=window.RigDevices.createCamera({
  preview:$('camCanvas'),
  onState:(s,msg)=>{
    if(s==='loading'){camChip('Waiting for permission…',true);status('Allow camera access in your browser, or click the camera toggle again to cancel. Idle animation continues.');}
    else if(s==='on'){camChip('Camera on');tracker.reset();}
    else camChip(toggleLabels.cam);
    if(s==='error'){auto.cam=false;syncToggle('cam');status(msg,true);cam.live=false;sync.publishTracking(liveTracking(),true);updateLiveTicker();}
    $('btnCalibrate').disabled=s!=='on';updateCamPreview();
  },
  onResults:(lm,video)=>{
    const aspect=video.videoWidth&&video.videoHeight?video.videoWidth/video.videoHeight:4/3;
    const out=lm?tracker.update(lm,performance.now()/1000,aspect):null;
    if(!out)return;   // a frame without a face (a hand, motion blur) holds the pose; 1.2 s without one hands back to idle
    Object.assign(cam,out);cam.live=true;lastTrackingAt=performance.now();
  }
});
function startCam(){
  if(OBS_MODE){status('Waiting for tracking from the main browser');return;}
  camera.start().catch(err=>{auto.cam=false;syncToggle('cam');camChip(toggleLabels.cam);status('Camera unavailable: '+err.message,true);updateLiveTicker();});
}
function stopCam(){
  if(OBS_MODE)return;
  camera.stop();cam.live=false;lastTrackingAt=0;sync.publishTracking(liveTracking(),true);
}
function updateCamPreview(){$('camPreview').hidden=OBS_MODE||!prefs.camPreview||!camera.active;}
$('camPreviewClose').addEventListener('click',()=>{prefs.camPreview=false;$('prefCamPreview').checked=false;savePrefs();updateCamPreview();});
function calibrationText(){$('calibrationStatus').textContent=prefs.calibration?'Calibrated to your neutral expression':'Using default calibration';$('btnClearCalibration').disabled=!prefs.calibration;}
calibrationText();
$('btnCalibrate').addEventListener('click',()=>{
  if(!camera.active)return;
  tracker.startCalibration();$('btnCalibrate').disabled=true;status('Look straight at the camera with a relaxed face for about a second…');
  // Average at least 1 s and 5 frames (slow PCs track fewer frames per second).
  const started=performance.now();
  const poll=setInterval(()=>{
    const n=tracker.samples?tracker.samples.length:0,elapsed=performance.now()-started;
    if(camera.active&&elapsed<6000&&(elapsed<1000||n<5))return;
    clearInterval(poll);
    const c=tracker.finishCalibration();$('btnCalibrate').disabled=!camera.active;
    if(!c){status('No face detected, so nothing was recorded',true);return;}
    prefs.calibration=c;savePrefs();calibrationText();tracker.reset();status('Neutral pose recorded ('+n+' frames)');
  },100);
});
$('btnClearCalibration').addEventListener('click',()=>{prefs.calibration=null;tracker.setCalibration(null);savePrefs();calibrationText();status('Calibration cleared');});

function micChip(text,busy){const el=$('tgMic');el.textContent=text;el.classList.toggle('busy',!!busy);}
const mic=window.RigDevices.createMic({onState:(s,msg)=>{
  if(s==='loading')micChip('Starting microphone…',true);
  else if(s==='on')micChip('Microphone on');
  else micChip(toggleLabels.mic);
  if(s==='error'){auto.mic=false;syncToggle('mic');status(msg,true);updateLiveTicker();}
}});
function startMic(){mic.start().catch(err=>{auto.mic=false;syncToggle('mic');micChip(toggleLabels.mic);status('Microphone unavailable: '+err.message,true);updateLiveTicker();});}
let meterAt=0;
function updateMeter(now){
  if(now-meterAt<50)return;meterAt=now;
  const m=$('micMeter');if(m.offsetParent===null)return;
  m.querySelector('.meter-level').style.width=(mic.active?Math.round(clamp(mic.raw*prefs.micGain,0,1)*100):0)+'%';
  m.querySelector('.meter-gate').style.left=Math.round(prefs.micGate*100)+'%';
}

// ---------- OBS sync ----------
const sync=window.ObsSync.create({obsMode:OBS_MODE,onTracking:receiveTracking,onState:receiveState,onModel:receiveModel,onStatus:renderObsStatus});
sync.setEnabled(prefs.obsSync);
function liveTracking(){
  const camOn=auto.cam&&cam.live&&performance.now()-lastTrackingAt<1200;
  const v={live:camOn,ax:cam.ax,ay:cam.ay,az:cam.az,eL:cam.eL,eR:cam.eR,mo:cam.mo,ex:cam.ex,ey:cam.ey,br:cam.br||0,mf:cam.mf||0};
  if(!camOn)Object.assign(v,{ax:0,ay:0,az:0,eL:1,eR:1,mo:0,ex:0,ey:0,br:0,mf:0});
  if(auto.mic&&mic.active)v.mic=micLevel;
  return RT.tracking(v);
}
function syncState(immediate){if(!OBS_MODE&&layers.length)sync.publishState({type:'state',settings:settingsSnapshot()},immediate);}
function receiveTracking(value){
  if(value===null){cam.live=false;return;}
  const clean=RT.tracking(value);if(!clean)return;
  if(clean.mic!==undefined){remoteMic=clean.mic;remoteMicAt=performance.now();}
  delete clean.mic;
  if(!auto.cam&&!OBS_MODE)return;
  Object.assign(cam,{br:0,mf:0},clean);lastTrackingAt=performance.now();
}
let pendingRemoteState=null,relayLoadingId=null;
function receiveState(state){
  if(!OBS_MODE||!state||!state.settings)return;
  if(!layers.length||state.settings.modelId!==modelId){pendingRemoteState=state;return;}
  try{applySettings(state.settings);}catch(err){}
}
async function loadFromRelay(id){
  relayLoadingId=id||'?';
  try{await loadModel(async()=>sync.fetchModel(),'OBS model',{relay:false});}
  finally{if(relayLoadingId===(id||'?'))relayLoadingId=null;}
}
function receiveModel(info){
  if(!OBS_MODE||!info||!info.id)return;
  if(info.id===(relayLoadingId||modelId))return;   // already showing / already loading this one
  loadFromRelay(info.id);
}
async function afterObsModel(){
  if(pendingRemoteState&&pendingRemoteState.settings.modelId===modelId){const s=pendingRemoteState;pendingRemoteState=null;receiveState(s);return;}
  if(sync.status.relay){try{const s=await sync.fetchState();if(s)receiveState(s);return;}catch(err){}}
  const path=RT.safeRelativePath(QUERY.get('settings')||'','.json');
  if(path){try{const r=await fetch(path);if(r.ok)applySettings(await r.json(),{anyModel:true});}catch(err){console.warn('settings:',err.message);}}
}
const obsUrl=location.origin+location.pathname.replace(/[^/]*$/,'')+'?obs=1';
$('obsUrl').value=location.protocol.startsWith('http')?obsUrl:' (shown when started with start_obs.bat)';
$('btnCopyObsUrl').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('obsUrl').value);status('OBS URL copied');}catch(err){$('obsUrl').select();status('URL selected. Press Ctrl+C to copy it');}});
$('obsChip').addEventListener('click',()=>{const s=document.querySelector('.sec[data-sec="obs"]');collapseSection(s,false);s.scrollIntoView({behavior:'smooth',block:'start'});});
function renderObsStatus(st){
  if(OBS_MODE)return;
  const chip=$('obsChip'),text=$('obsStatus');
  chip.hidden=!st.relay;
  if(!st.relay){text.textContent='Not connected to the OBS relay. Open the page through start_obs.bat (python obs_server.py) to mirror this browser to OBS.';return;}
  if(!st.features.includes('model')){chip.textContent='OBS';chip.className='obs-chip busy';text.textContent='This obs_server.py is outdated; replace it to sync expressions and models (face tracking only for now).';return;}
  const viewers=st.viewers|0;
  chip.className='obs-chip'+(!st.enabled?' off':st.uploading?' busy':'');
  chip.textContent=!st.enabled?'OBS sync off':st.uploading?'OBS sending…':viewers?'OBS ● live '+viewers:'OBS waiting';
  text.textContent=st.error?st.error:!st.enabled?'Sync is off. OBS shows the last state sent.':
    'Connected to the relay · '+viewers+' OBS view(s) · '+(st.uploaded&&st.uploaded===modelId?'this character sent':layers.length?'waiting to send the character':'Loading character…');
}

// ---------- animation ----------
let blinkT=-1,blinkVariant=1,blinkBounceStarted=false,irisBounceT=-1,nextBlink=performance.now()+1800,blinkStart=0;
const rnd={ax:0,ay:0,az:0,bd:0,ex:0,ey:0,head:null,headAt:0};let nextRnd=0;
// Idle eye darts; speech beats (a small nod on each stressed syllable); head-turn speed.
const gaze={x:0,y:0,next:0},beat={v:0,armed:true};let lastYaw=0;
let talkOn=false,talkV=0,talkTgt=0,nextTalkState=0,nextSyl=0;
// Painted vowel mouths: current weights, the target, and which vowels this character has.
let visemes={a:1,i:0,u:0,e:0,o:0},visemeTarget={a:1,i:0,u:0,e:0,o:0},visemeSet=new Set(),visemeOrder=[];
// Rough vowel frequency in Japanese speech, for Auto talk's made-up syllables.
const VOWEL_MIX=[['a',.3],['i',.2],['u',.15],['e',.13],['o',.22]];
function randomVowel(){let r=Math.random();for(const [v,p] of VOWEL_MIX){if((r-=p)<0)return v;}return 'a';}
function oneVowel(v){return {a:0,i:0,u:0,e:0,o:0,[v]:1};}

function fadeAlpha(L,e,alt){
  if(L.companion)return (L.kind==='ear'?e.neko:e.tail)>.5?1:0;
  if(e.replaced?.has(L.bn))return 0;   // an alternate hairstyle/outfit is on
  const weights=Motion.mouthWeights(e.mouthOpen,e.mouthTeeth,e.mouthRound,e.mouthExpressive);
  if(L.expression)return weights[L.expression];
  if(!L.fade)return 1;
  if(L.bn==='toggle sparkle aura'||L.bn==='toggle floating hearts')return e.emotePulse??1;
  const eyeOpen=L.side==='L'?e.eyeOpenL:e.eyeOpenR;
  const openAlpha=smooth((eyeOpen-(0.10+e.eyeEase*0.45))/0.15);
  if(L.toggleEye)return L.toggleEye==='open'?openAlpha:1;
  const closedOverlay=e.eyeToggles?.[L.side]?.closed||0;
  if(L.fade==='eyeOpen')return openAlpha*(1-closedOverlay)*(L.bn==='irides'?1-(e.eyeToggles?.[L.side]?.open||0):1);
  if(L.fade==='eyeClose'||L.fade==='eyeClose2'){
    const alternate=blinkVariant===2&&alt[L.side];
    if((L.fade==='eyeClose2')!==alternate)return 0;
    const v=L.side==='L'?e.eyeOpenL:e.eyeOpenR;return (1-smooth((v-(0.10+e.eyeEase*0.45))/0.15))*(1-closedOverlay);
  }
  if(L.fade==='viseme')return Motion.mouth(e.mouthOpen,e.mouthForm,e.mouthRound,e.mouthEase).alpha*weights.source*(e.visemeAlpha?.[L.bn.slice(6)]||0);
  if(L.fade==='mouthOpen')return Motion.mouth(e.mouthOpen,e.mouthForm,e.mouthRound,e.mouthEase).alpha*weights.source*(1-(e.visemeCover||0));
  if(L.fade==='mouthClose')return (1-Motion.mouth(e.mouthOpen,e.mouthForm,e.mouthRound,e.mouthEase).alpha)*weights.source;
  return 1;
}
const FRONTISH={'front hair':1,'side hair':1,'ahoge':1};
// Symbols drawn beside the head bob gently with Idle instead of sitting glued to it.
const FLOATING=/^toggle (sparkle aura|floating hearts|music notes|question mark|exclamation|zzz)$/;
function deform(L,e){
  const b=L.base,o=L.cur,n=b.length;
  const isHead=L.group==='head',isBody=L.group==='body'&&!L.companion;
  const fw=A.face.x1-A.face.x0,fh=A.face.y1-A.face.y0,shoulderDrop=Math.min(fh*1.3,(CH-NP.cy)*.85);
  const az=e.angleZ*Motion.ROLL;
  const bn=L.bn,eyeSide=L.side,EA=eyeSide==='L'?A.eyeL:(eyeSide==='R'?A.eyeR:null);
  const vOpen=eyeSide==='L'?e.eyeOpenL:e.eyeOpenR;
  const mo=e.mouthOpen,mHalfW=(A.mouth.x1-A.mouth.x0)/2,mouthShape=Motion.mouth(mo,e.mouthForm,e.mouthRound,e.mouthEase);
  const nS=L.strands?L.strands.length:0;
  const bcx=L.x+L.w/2,bcy=L.y+L.h/2;
  const physOn=nS&&auto.phys&&L.su;
  for(let k=0;k<n;k+=2){
    let x=b[k],y=b[k+1];
    const vi=k>>1;
    if(L.companion){[x,y]=companionPoint(L,x,y,e);}
    if(bn==='cat ears'){const pivot=L.y+L.h*.94,side=L.side==='L'?-1:1;const p=Motion.earPoint((x-bcx)/L.w,(y-pivot)/L.h,L.w,L.h,side,(e['ear'+(side<0?'L':'R')]||0)*e.earMotion);x=bcx+p[0];y=pivot+p[1];}
    // --- closed-eye / mouth scale ---
    if(EA&&(bn==='eye_close'||bn==='eye_close2')){
      const sE=eyeSide==='L'?e.eyeScaleL:e.eyeScaleR;
      const point=RT.closedEyePoint(x,y,EA,L.closeY ?? baseAnchors['eye'+eyeSide].closeY,sE,e.eyeCY*14*FS,e.eyeCAng*.3*(eyeSide==='L'?1:-1)+(eyeSide==='L'?e.eyeTiltL:e.eyeTiltR)*Math.PI/180);
      x=point[0];y=point[1];
    }
    if(!L.expression&&(bn==='mouth_open'||bn==='mouth_close')){
      const sM=e.mouthScale;
      if(sM!==1){x=A.mouth.cx+(x-A.mouth.cx)*sM;y=A.mouth.cy+(y-A.mouth.cy)*sM;}
    }
    // --- local features ---
    if(FLOATING.test(L.bn))y+=e.emoteLift||0;
    if(L.toggleEye==='open'&&EA)y=EA.closeY+(y-EA.closeY)*(1-.85*(1-vOpen));
    if(L.fade==='eyeOpen'&&EA){
      if(bn==='irides'){
        const isc=e.irisScale,ibx=e.irisBounceX||1,iby=e.irisBounceY||1;
        x=EA.icx+(x-EA.icx)*isc*ibx;y=EA.icy+(y-EA.icy)*isc*iby;
        x+=e.eyeX*11*FS;y+=e.eyeY*6*FS;
        const tl=smooth((0.32-vOpen)/0.32);   // iris stays round until nearly closed
        y=EA.closeY+(y-EA.closeY)*(1-0.80*tl);
      }else{
        y=EA.closeY+(y-EA.closeY)*(1-0.85*(1-vOpen));   // lid compression
      }
    }
    if(bn==='eyebrow'){
      y+=(-e.brow*9+(1-vOpen)*3.5)*FS;
      const th=(eyeSide==='L'?(e.browAngL+e.browAngSym):(e.browAngR-e.browAngSym))*0.30;
      if(th){const ct=Math.cos(th),st=Math.sin(th),rx=x-bcx,ry=y-bcy;x=bcx+rx*ct-ry*st;y=bcy+rx*st+ry*ct;}
    }
    if(L.fade==='mouthOpen'&&!L.expression){
      x=A.mouth.cx+(x-A.mouth.cx)*mouthShape.width;
      y=A.mouth.y0+(y-A.mouth.y0)*mouthShape.height;
      const q=Math.pow(Math.abs(x-A.mouth.cx)/(mHalfW+4),1.5);
      y-=e.mouthForm*6*FS*(q-0.35);
    }
    if(L.fade==='viseme'){
      x=A.mouth.cx+(x-A.mouth.cx)*e.mouthScale;
      y=A.mouth.y0+(y-A.mouth.y0)*e.mouthScale*(.55+.45*mo);
    }
    if(L.expression){
      const m=L.mouthAnchor,cx=m.cx+(A.mouth.cx-baseAnchors.mouth.cx),cy=m.cy+(A.mouth.cy-baseAnchors.mouth.cy);
      x=cx+(x-m.cx)*e.mouthScale*(L.expression==='grin'?1+.08*e.mouthForm:1);
      y=cy+(y-m.cy)*e.mouthScale*(.25+.75*mo)*(1+.25*e.jawDrop)+e.jawDrop*3*FS;
    }
    if(L.fade==='mouthClose'){
      x=A.mouth.cx+(x-A.mouth.cx)*(1+.08*e.mouthForm);
      y-=e.mouthForm*3*FS*(Math.pow(Math.abs(x-A.mouth.cx)/(mHalfW+4),1.5)-.35);
      y+=e.mouthCY*14*FS;
      const thM=e.mouthCAng*0.35;
      if(thM){const ct=Math.cos(thM),st=Math.sin(thM),rx=x-A.mouth.cx,ry=y-A.mouth.cy;x=A.mouth.cx+rx*ct-ry*st;y=A.mouth.cy+rx*st+ry*ct;}
    }
    if(bn==='face'&&y>A.mouth.cy)y+=(mo*6+e.jawDrop*10)*FS*smooth((y-A.mouth.cy)/Math.max(1,A.face.y1-A.mouth.cy));
    // --- head transform: roll about the neck pivot; the neck and collar bend from the jaw down ---
    const nw=isBody?Motion.collarWeight(b[k],b[k+1],NP,A.face.y1,fw*.22,fh*.3,.35):0;   // the Neck pivot handle sets where the bend eases off
    const rw=isHead?1:nw,hw=isHead?1:Math.max(nw,L.group==='body'?0.16:0);   // body subtly follows head XY
    if(rw>0)[x,y]=Motion.rollPoint(x,y,NP,A.face.y1,az,rw,fh);
    if(hw>0){
      const dd=L.depth;
      x+=hw*FS*(e.angleX*(14+40*(dd-1))+e.angleX*(NP.cy-Math.min(y,A.face.y1))*0.028);
      y+=hw*FS*(-e.angleY*(9+30*(dd-1))-e.angleY*(dd-1)*(y-FC.y)*0.05);
    }
    // Head-led nod: a shared facial transform; neck blends down to still shoulders.
    const nodWeight=isHead?1:nw;
    y=Motion.nodPoint(y,NP.cy,A.face.y1-A.face.y0,e.nodPose||{head:0,body:0,squash:1},nodWeight);
    // --- shoulders lift and the torso turns; where the collar follows the head it skips the turn ---
    if(isBody){
      y-=Motion.shoulderLift(b[k],b[k+1],NP,fw*.25,fw*.8,shoulderDrop,e.shoulderL||0,e.shoulderR||0,bn==='handwear');
      x+=Motion.turnShift(b[k],b[k+1],NP.cx,fw*.9,NP.cy+fh*1.4,fh*.5,e.turn||0)*(1-nw);
    }
    // --- a hat or bow trails the head ---
    if(bn==='headwear'){const w=Motion.hatWeight(b[k],b[k+1],{cx:A.face.cx,cy:A.face.y0},fh*.5);x+=(e.hatX||0)*w;y+=(e.hatY||0)*w;}
    // --- breathing ---
    y-=(L.group==='body'?e.breath*2.0:e.breathHead*1.6)*FS;
    if(bn==='topwear'&&y<CHEST.cy)y-=e.breath*2.2*FS*smooth((CHEST.cy-y)/(CHEST.ry*2));   // shoulders rise
    if(bn==='topwear')x=NP.cx+(x-NP.cx)*(1+e.breath*0.003);
    // --- bust jiggle ---
    if(bn==='topwear'){
      const gx=(x-CHEST.cx)/CHEST.rx,gy=(y-(CHEST.cy+e.bustY*70*FS))/CHEST.ry;
      y+=bounce.dy*e.bust*Math.exp(-gx*gx-gy*gy);
    }
    // --- arms ---
    if(bn==='handwear'){
      const w=smooth((y-L.y)/L.h*1.15);
      y-=e.armY*30*FS*w;y+=e.armPos*40*FS;x+=e.armY*6*FS*w*(x<NP.cx?1:-1);
    }
    // --- bang blocks ---
    if(L.bw){const m=Math.pow(L.su[vi],1.4)*22*FS;x+=(e.bangL*L.bw[vi*3]+e.bangC*L.bw[vi*3+1]+e.bangR*L.bw[vi*3+2])*m;}
    // --- strand physics: stiff root, fluffy tips (front hair / accessories have their own params) ---
    if(physOn){
      const binding=L.bindings[vi];
      const dx=hairShift(L,binding,e);
      x+=dx;y+=Math.abs(dx)*0.12+hairLift(L,binding,e);
    }
    o[k]=x;o[k+1]=y;
  }
  // Torso parent applied after local neck/shoulder/head motion, with planted lower points.
  if(e.torso)for(let k=0;k<n;k+=2){
    const p=Motion.torsoPoint(o[k],o[k+1],L.companion?o[k+1]:b[k+1],e.torsoGeometry,e.torso);
    o[k]=p[0];o[k+1]=p[1];
  }
  // A common follow-through transform keeps all facial parts and attachments aligned.
  for(let k=0;k<n;k+=2){
    const sourceY=L.companion?o[k+1]:b[k+1],ground=1-smooth((sourceY-NP.cy)/Math.max(1,CH-NP.cy));
    o[k]+=(e.followX||0)*ground;o[k+1]+=(e.followY||0)*ground;
  }
}
function updateSprings(e,t,dt,wind){
  const headDX=e.angleX*14*FS+Math.sin(e.angleZ*Motion.ROLL)*(NP.cy-FC.y)+(e.torso?Motion.torsoPoint(FC.x,FC.y,FC.y,e.torsoGeometry,e.torso)[0]-FC.x:0);
  const headDY=(e.nodPose?.head||0)*(A.face.y1-A.face.y0)-e.angleY*9*FS;
  const bodyDX=e.torso?Motion.torsoPoint(NP.cx,NP.cy,NP.cy,e.torsoGeometry,e.torso)[0]-NP.cx:0;
  for(const L of layers){
    if(!L.spr)continue;
    const base=L.group==='head'?headDX:bodyDX,windScale=L.phys==='sway'&&L.group!=='head'?2.5:1;
    for(const sp of L.spr){
      const w=wind?(1.8*Math.sin(t*0.8+sp.phase)+1.0*Math.sin(t*1.9+sp.phase*2.3))*windScale:0;
      const txv=base+w*FS*(1+e.motionBounce*1.8);
      RT.spring(sp.stiff,txv,70,9-e.motionBounce*2,dt);sp.stiff.dx=-(sp.stiff.x-txv)*2.2;
      RT.spring(sp.soft,txv,24,4-e.motionBounce*1.5,dt);sp.soft.dx=-(sp.soft.x-txv)*3.0;
      const tyv=L.group==='head'?headDY:(e.nodPose?.body||0)*(A.face.y1-A.face.y0);
      RT.spring(sp.liftStiff,tyv,120,14,dt);sp.liftStiff.dx=sp.liftStiff.x-tyv;
      RT.spring(sp.liftSoft,tyv,46,7,dt);sp.liftSoft.dx=(sp.liftSoft.x-tyv)*1.3;
    }
  }
  const bustTgt=(e.breath*3.0-e.angleY*6.0+e.body*4.0)*FS;
  RT.spring(bounce,bustTgt,140,4.2,dt);bounce.dy=auto.phys?-(bounce.x-bustTgt)*3.0:0;
  const bx=(e.body*7+e.angleX*3)*FS,by=(-e.angleY*3+(wind?Math.sin(t*2.2)*1.4:0))*FS;
  RT.spring(motion.x,bx,52,8-e.motionBounce*2,dt);RT.spring(motion.y,by,64,8-e.motionBounce*2,dt);
  e.followX=auto.phys?(motion.x.x-bx)*e.motionBounce:0;
  e.followY=auto.phys?((motion.y.x-by)+(wind?Math.sin(t*2.2)*2.5*FS:0))*e.motionBounce:0;
  for(const [side,offset] of [['L',0],['R',1.6]]){
    const twitch=wind?Math.pow(Math.max(0,Math.sin(t*.83+offset)),18)*.24+Math.sin(t*1.3+offset)*.025:0;
    RT.spring(motion['ear'+side],twitch-e.angleZ*.18-e.angleX*.08,90,6,dt);
    e['ear'+side]=auto.phys?motion['ear'+side].x:0;
  }
  // Headwear lags where the crown goes (a roll moves the crown further than the face centre).
  const fh=A.face.y1-A.face.y0,hatX=e.angleX*14*FS+Math.sin(e.angleZ*Motion.ROLL)*(NP.cy-A.face.y0),limit=fh*.12;
  RT.spring(motion.hatX,hatX,42,5-e.motionBounce*1.5,dt);RT.spring(motion.hatY,headDY,55,5,dt);
  e.hatX=auto.phys?clamp((motion.hatX.x-hatX)*2.6+(wind?fh*(.025*Math.sin(t*1.3)+.01*Math.sin(t*3.1)):0),-limit,limit)*e.accAmp:0;
  e.hatY=auto.phys?clamp((motion.hatY.x-headDY)*2,-limit,limit)*e.accAmp:0;
  RT.spring(motion.tail,e.body*.5+e.angleX*.28,22,4,dt);
  e.tailPhase=t*1.8;e.tailSway=auto.phys?clamp((wind?.8:0)+(motion.tail.x-e.body*.5-e.angleX*.28)*3,-1.8,1.8)*e.tailMotion:0;
}
// Still pose used while editing anchors: handles line up with the drawing.
function restPose(dt){
  const e=Object.assign({},T,{angleX:0,angleY:0,angleZ:0,body:0,eyeX:0,eyeY:0,irisBounceX:1,irisBounceY:1,breath:0,breathHead:0});
  if(blinkEditor){e.eyeOpenL=blinkView==='both'||blinkView==='L'?0:1;e.eyeOpenR=blinkView==='both'||blinkView==='R'?0:1;}
  else{if(anchorPreview==='L')e.eyeOpenL=0;if(anchorPreview==='R')e.eyeOpenR=0;}
  updateSprings(e,0,dt,false);e.followX=e.followY=e.earL=e.earR=e.tailSway=e.hatX=e.hatY=0;
  return e;
}
function animate(now,dt){
  if(anchorMode)return restPose(dt);
  const t=now/1000;
  const tgt=Object.assign({},T);
  const camLive=(auto.cam||OBS_MODE)&&cam.live&&performance.now()-lastTrackingAt<1200;
  const turnSpeed=Math.abs(cur.angleX-lastYaw)/Math.max(dt,1e-3);lastYaw=cur.angleX;
  if(camLive){
    tgt.angleX=clamp(cam.ax,-1,1);tgt.angleY=clamp(cam.ay,-1,1);tgt.angleZ=clamp(cam.az,-1,1);
    tgt.eyeOpenL=Math.min(tgt.eyeOpenL,cam.eL);tgt.eyeOpenR=Math.min(tgt.eyeOpenR,cam.eR);
    tgt.eyeX=clamp(cam.ex,-1,1);tgt.eyeY=clamp(cam.ey,-1,1);
    tgt.mouthOpen=Math.max(tgt.mouthOpen,cam.mo);
    tgt.brow=clamp(tgt.brow+(cam.br||0),-1,1);tgt.mouthForm=clamp(tgt.mouthForm+(cam.mf||0),-1,1);
    // Face yaw drives torso yaw later, with its own lag rather than becoming body roll.
  }
  if(!camLive&&auto.mouse&&mouse.in){tgt.angleX=clamp(mouse.x*0.9,-1,1);tgt.angleY=clamp(-mouse.y*0.7,-1,1);tgt.eyeX=clamp(mouse.x*1.2,-1,1);tgt.eyeY=clamp(-mouse.y*0.8,-1,1);}
  if(auto.idle&&!camLive){
    const life=(1+T.motionBounce*.75)*(1-Math.min(1,Math.max(T.nodDepth,T.swayDepth))*.55);
    const head=Motion.idleHead(t);
    tgt.angleX+=head.x*life;tgt.angleY+=head.y*life;tgt.angleZ+=head.z*life;
    // Eyes rest on one point, then dart to the next (a saccade), most often back to centre.
    if(now>gaze.next){const away=Math.random()<.55;gaze.x=away?(Math.random()*2-1)*.22:0;gaze.y=away?(Math.random()*2-1)*.1:0;gaze.next=now+900+Math.random()*2400;}
    if(!(auto.mouse&&mouse.in)){tgt.eyeX+=gaze.x;tgt.eyeY+=gaze.y;}
    tgt.body+=(Motion.idlePose(t).roll+.06*Math.sin(t*.71)) * life*T.bodyMotion;
  }
  if(auto.rand&&!camLive){
    // The eyes find the new point first; the head follows ~130 ms later (eye-head coordination).
    if(now>nextRnd){nextRnd=now+1400+Math.random()*2600;rnd.headAt=now+130;
      rnd.head={ax:(Math.random()*2-1)*0.55,ay:(Math.random()*2-1)*0.40,az:(Math.random()*2-1)*0.35,bd:(Math.random()*2-1)*0.30};
      rnd.ex=(Math.random()*2-1)*0.60;rnd.ey=(Math.random()*2-1)*0.35;}
    if(rnd.head&&now>=rnd.headAt){Object.assign(rnd,rnd.head);rnd.head=null;}
    tgt.angleX=clamp(tgt.angleX+rnd.ax,-1,1);tgt.angleY=clamp(tgt.angleY+rnd.ay,-1,1);
    tgt.angleZ=clamp(tgt.angleZ+rnd.az,-1,1);tgt.body=clamp(tgt.body+rnd.bd,-1,1);
    tgt.eyeX=clamp(tgt.eyeX+rnd.ex,-1,1);tgt.eyeY=clamp(tgt.eyeY+rnd.ey,-1,1);
  }
  if(auto.idle&&!camLive)motion.nodTime=(motion.nodTime+dt*cur.nodSpeed)%2;   // sway runs at half speed
  const nod=auto.idle&&!camLive?Motion.nodPose(motion.nodTime,cur.nodDepth,1,cur.swayDepth):{head:0,body:0,squash:1,pitch:0,tilt:0};
  tgt.angleY+=nod.pitch;tgt.angleZ+=nod.tilt;
  if(!camLive){
    tgt.angleZ+=Motion.TURN_TILT*tgt.angleX;
    tgt.angleY-=Math.min(.12,.05*turnSpeed);   // a head turn travels on a dipping arc, not a flat slide
  }
  const remoteMicLive=OBS_MODE&&performance.now()-remoteMicAt<500;
  if(auto.talk&&!camLive&&!auto.mic&&!remoteMicLive){
    if(now>nextTalkState){talkOn=!talkOn;nextTalkState=now+(talkOn?1200+Math.random()*2200:600+Math.random()*1800);}
    if(talkOn&&now>nextSyl){nextSyl=now+70+Math.random()*110;talkTgt=Math.random()<0.25?0.04:0.25+Math.random()*0.75;visemeTarget=oneVowel(randomVowel());}
    if(!talkOn)talkTgt=0;
    talkV+=(talkTgt-talkV)*Math.min(1,dt*22);
    tgt.mouthOpen=talkV;
    tgt.mouthRound=Math.max(tgt.mouthRound,talkOn?.45+.4*Math.sin(t*8.2):0);
  }
  if(auto.blink&&!camLive&&!activePreset){
    if(turnSpeed>5&&blinkT<0&&now-blinkStart>700)nextBlink=Math.min(nextBlink,now);   // animators blink on big turns
    if(blinkT<0&&now>nextBlink){blinkT=0;blinkStart=now;blinkBounceStarted=false;blinkVariant=hasEyeClose2&&Math.random()<0.2?2:1;nextBlink=now+1600+Math.random()*3800;if(Math.random()<0.18)nextBlink=now+280;}
    if(blinkT>=0){blinkT+=dt;
      const d=blinkT,hold=blinkVariant===2?.65:.04;let v=Motion.blink(d,hold);
      if(v<0){v=1;blinkT=-1;}
      else if(d>.09+hold&&!blinkBounceStarted&&v>0.12){blinkBounceStarted=true;irisBounceT=0;}
      tgt.eyeOpenL=Math.min(tgt.eyeOpenL,v);tgt.eyeOpenR=Math.min(tgt.eyeOpenR,v);
    }
  }
  if(irisBounceT>=0){irisBounceT+=dt;if(irisBounceT>0.52)irisBounceT=-1;}
  if(auto.mic&&mic.active)tgt.mouthOpen=Math.max(tgt.mouthOpen,micLevel);
  if(remoteMicLive)tgt.mouthOpen=Math.max(tgt.mouthOpen,remoteMic);
  if(visemeSet.size){
    if(auto.mic&&mic.active&&micLevel>.12){const f=mic.frame();const w=f&&RigVisemes.classify(f.samples,f.rate);if(w)visemeTarget=w;}
    else if(camLive)visemeTarget=RigVisemes.fromShape(cam.mo,cam.mf||0,0);
    visemes=RigVisemes.blend(visemes,visemeTarget,dt,18);
  }
  // Speech beat: each strong syllable onset dips the head a little (Live2D motion tips: move with speech).
  if(tgt.mouthOpen>.45&&beat.armed){beat.v=1;beat.armed=false;}else if(tgt.mouthOpen<.15)beat.armed=true;
  beat.v*=Math.exp(-dt*7);
  if(!camLive)tgt.angleY-=.06*beat.v*T.bodyMotion;
  const kSmooth=1-Math.exp(-dt*14);
  for(const k in cur){tgt[k]=RT.clamp(tgt[k],...parameterRanges[k]);
    if(['mouthExpressive','neko','tail','furPalette','tailSide'].includes(k)){cur[k]=tgt[k];continue;}
    if(k==='mouthOpen'){cur[k]=Motion.response(cur[k],tgt[k],dt,tgt.mouthResponse);continue;}
    const response=k==='eyeOpenL'||k==='eyeOpenR'?1-Math.exp(-dt*(tgt[k]<cur[k]?55:28)):k==='eyeX'||k==='eyeY'?1-Math.exp(-dt*32):kSmooth;cur[k]+=(tgt[k]-cur[k])*response;}
  const e=Object.assign({},cur);
  e.visemes=visemes;e.visemeCover=0;for(const v of visemeSet)e.visemeCover+=visemes[v]||0;
  e.visemeAlpha=RigVisemes.stackAlpha(visemes,visemeOrder);
  e.nodPose=camLive?{head:-e.angleY*e.nodDepth*.08,body:0,squash:1}:nod;
  e.irisBounceX=1;e.irisBounceY=1;
  if(irisBounceT>=0){
    const p=irisBounceT/0.52,damp=Math.exp(-2.2*p);
    const scale=1+0.18*Math.sin(p*Math.PI*4.0)*damp,squash=0.10*Math.sin(p*Math.PI*4.0+Math.PI/2)*damp;
    e.irisBounceX=scale*(1+squash);e.irisBounceY=scale*(1-squash);
  }
  camPhysScale+=((camLive?0.85:1)-camPhysScale)*Math.min(1,dt*4);   // calmer hair while tracking
  e.physAmp*=camPhysScale;e.fhAmp*=camPhysScale;
  e.emotePulse=auto.idle?.82+.18*Math.sin(t*2.2):1;
  e.emoteLift=auto.idle?Math.sin(t*1.6)*(A.face.y1-A.face.y0)*.012:0;
  e.breath=auto.idle?Motion.breath(t)*e.breathing:0;
  e.breathHead=auto.idle?Motion.breath(t-.35)*e.breathing:0;   // head follows chest with a lag
  // Shrug lifts both shoulders; breathing a little, a tilt the shoulder it leans towards, and Idle
  // rolls them out of step. Turning the head turns the torso a little.
  const roll=auto.idle&&!camLive?.16:0,lean=v=>.5*Math.pow(Math.max(0,v),1.5),lift=20*FS;
  e.shoulderL=lift*(e.shrug+e.bodyMotion*(.2*e.breath+lean(-e.angleZ)+roll*Math.sin(t*.53)));
  e.shoulderR=lift*(e.shrug+e.bodyMotion*(.2*e.breath+lean(e.angleZ)+roll*Math.sin(t*.53+2.1)));
  e.turn=0; // Turning now belongs to the torso parent, shared with the head and arms.
  const g=e.torsoGeometry={cx:NP.cx,shoulderY:NP.cy,hipY:Math.min(CH,NP.cy+(A.face.y1-A.face.y0)*1.65),width:A.face.x1-A.face.x0,height:A.face.y1-A.face.y0,footY:CH};
  const speech=(auto.talk||auto.mic||camLive)?e.mouthOpen:0;
  const target={yaw:clamp(e.bodyTurn+e.bodyMotion*e.angleX*.65,-1,1),roll:clamp(e.body+e.bodyMotion*e.angleZ*.35,-1,1),lift:e.bodyMotion*(e.angleY*.25+speech*.12),shift:auto.idle&&!camLive?Motion.idlePose(t).shift*e.weightShift:0};
  if(auto.phys){
    RT.spring(motion.torsoYaw,target.yaw,32,9,dt);RT.spring(motion.torsoRoll,target.roll,38,9,dt);RT.spring(motion.torsoLift,target.lift,30,9,dt);RT.spring(motion.torsoShift,target.shift,22,9,dt);
    e.torso={yaw:motion.torsoYaw.x,roll:motion.torsoRoll.x,lift:motion.torsoLift.x,shift:motion.torsoShift.x};
  }else e.torso=target;
  updateSprings(e,t,dt,auto.idle);
  return e;
}

// ---------- render loop ----------
function companionPoint(L,x,y,e){
  const fw=A.face.x1-A.face.x0,fh=A.face.y1-A.face.y0;
  if(L.kind==='ear'){
    const width=fw*.34*e.earSize,height=fw*.50*e.earSize;
    const p=Motion.earPoint((x-64)/128,(y-151)/151,width,height,L.earSide,
      .14+(e[L.earSide<0?'earL':'earR']||0)*e.earMotion);
    return [A.face.cx+L.earSide*fw*.42*e.earSpread+p[0],L.crown+fh*.12+e.earLift*fh+p[1]];
  }
  const side=e.tailSide<0?-1:1;
  const p=Motion.tailPoint((x-64)/128,clamp((380-y)/365,0,1),fh*1.25*e.tailSize,fw*.26*e.tailSize,side,e.tailPhase||0,e.tailSway||0);
  const torso=layers.find(part=>part.bn==='topwear');
  const rootY=torso?Math.min(NP.cy+fh*.8,torso.y+torso.h*.65):NP.cy+fh*.6;
  return [NP.cx+side*fw*.28+e.tailX*fw+p[0],rootY+e.tailY*fh+p[1]];
}
function updateStageBounds(){
  if(recorder?.active)return; // Keep recording resolution fixed until capture finishes.
  const key=JSON.stringify([T.neko,T.tail,T.earSize,T.earSpread,T.earLift,T.earMotion,T.tailSize,T.tailX,T.tailY,T.tailSide,T.tailMotion,T.nodDepth,T.accAmp,A.face,NP]);
  if(key===renderBox.key)return;
  let left=0,top=0,right=CW,bottom=CH;
  const fh=A.face.y1-A.face.y0,bodyGeometry={cx:NP.cx,shoulderY:NP.cy,hipY:Math.min(CH,NP.cy+fh*1.65),width:A.face.x1-A.face.x0,height:fh,footY:CH};
  function includeTorso(p,sourceY,pad=0){
    for(const roll of [-1,1])for(const yaw of [-1,1])for(const lift of [-1,1])for(const shift of [-2,2]){
      const q=Motion.torsoPoint(p[0],p[1],sourceY,bodyGeometry,{roll,yaw,lift,shift});
      left=Math.min(left,q[0]-pad);top=Math.min(top,q[1]-pad);right=Math.max(right,q[0]+pad);bottom=Math.max(bottom,q[1]+pad);
    }
  }
  if(T.nodDepth>0){
    const fh=A.face.y1-A.face.y0,headTop=Math.min(...layers.filter(L=>L.group==='head'&&!L.companion).map(L=>L.y));
    top=Math.min(0,headTop-fh*.14*T.nodDepth);bottom+=fh*.025*T.nodDepth;
  }
  // A full head tilt either way (camera, Tilt or sway) stays on the canvas. Art already cut
  // off at the border gains nothing from extra space (its cut would show), so it is skipped.
  const cut=Math.max(4,Math.max(CW,CH)*.01);   // matte edges sit a few px inside the border
  for(const L of layers)if(L.group==='head'&&L.outline&&!L.outline.some(([x,y])=>x<=cut||y<=cut||x>=CW-cut))for(const a of [-Motion.ROLL,Motion.ROLL])for(const [x,y] of L.outline){
    if(y>=CH-cut)continue;   // bust-up crops cut hair at the bottom; it barely moves there
    const p=Motion.rollPoint(x,y,NP,A.face.y1,a,1,A.face.y1-A.face.y0),pad=L.bn==='headwear'?(A.face.y1-A.face.y0)*.12*T.accAmp:0;   // hat lag limit
    includeTorso(p,y,pad);
  }
  const pose={...T,earL:0,earR:0,tailPhase:0,tailSway:0};
  for(const L of layers.filter(L=>L.companion&&(L.kind==='ear'?T.neko:T.tail)>.5)){
    const pad=(L.kind==='ear'?20*FS+(A.face.y1-A.face.y0)*.14*T.nodDepth:(A.face.y1-A.face.y0)*.38*T.tailSize*T.tailMotion+16*FS);
    for(const twitch of L.kind==='ear'?[-.8,0,.8]:[0]){
      pose.earL=pose.earR=twitch;
      for(let j=0;j<=16;j++)for(const x of [0,L.w]){
        const p=companionPoint(L,x,L.h*j/16,pose);
        includeTorso(p,p[1],pad);
      }
    }
  }
  // Respect the same device limits as the PSD loader, including added transparent space.
  const viewport=gl.getParameter(gl.MAX_VIEWPORT_DIMS),max=gl.getParameter(gl.MAX_TEXTURE_SIZE);
  const ew=right-left-CW,eh=bottom-top-CH,b=CW*eh+CH*ew,budget=Math.max(0,24000000-CW*CH);
  const areaScale=b>0?Math.min(1,2*budget/(b+Math.sqrt(b*b+4*ew*eh*budget))):1;
  const sx=Math.min(areaScale,Math.max(0,Math.min(viewport[0],max)-CW)/Math.max(1,ew));
  const sy=Math.min(areaScale,Math.max(0,Math.min(viewport[1],max)-CH)/Math.max(1,eh));
  renderBox={left:Math.floor(left*sx),top:Math.floor(top*sy),w:Math.ceil(CW+(right-left-CW)*sx),h:Math.ceil(CH+(bottom-top-CH)*sy),key};
  cv.width=renderBox.w;cv.height=renderBox.h;
  $('overlay').setAttribute('viewBox',[renderBox.left,renderBox.top,renderBox.w,renderBox.h].join(' '));fit();
}
function updateCompanionArt(e){
  const palette=Math.round(e.furPalette);if(palette===motion.palette)return;
  const rgb=Motion.palette(currentRig,palette);
  for(const L of layers.filter(L=>L.companion)){
    const image=Motion.art(L.kind,rgb);gl.activeTexture(gl.TEXTURE0);
    const old=L.tex;L.tex=renderer.texture(image);gl.deleteTexture(old);
  }
  motion.palette=palette;
}
function render(e){
  updateStageBounds();updateCompanionArt(e);
  // Observable diagnostics for browser checks; no imagery or device data is logged.
  const closed=e.eyeOpenL<0.1;
  if(closed&&!closedOnPreviousFrame)blinkCount++;
  closedOnPreviousFrame=closed;
  cv.dataset.blinkCount=String(blinkCount);
  cv.dataset.eyeOpen=String(Math.round(e.eyeOpenL*100)/100);
  cv.dataset.mouthOpen=String(Math.round(e.mouthOpen*100)/100);
  cv.dataset.headTurn=String(Math.round(e.angleX*100)/100);

  const alt={L:false,R:false};
  for(const L of layers)if(L.fade==='eyeClose2'&&L.side&&L.visible&&L.opacity>0)alt[L.side]=true;
  e.eyeToggles={L:{open:0,closed:0},R:{open:0,closed:0}};
  for(const L of layers)if(L.visible&&L.toggleEye&&L.side)e.eyeToggles[L.side][L.toggleEye]=Math.max(e.eyeToggles[L.side][L.toggleEye],L.opacity);
  e.replaced=new Set(layers.filter(L=>L.replaces&&L.visible).flatMap(L=>L.replaces));
  const items=[],masks=[];
  for(const L of layers){
    moveHairShading(L,e);
    const isMask=L.bn==='eyewhite'&&!!L.side;
    const alpha=L.visible?fadeAlpha(L,e,alt)*L.opacity:0;
    if(alpha<0.004&&!isMask)continue;
    deform(L,e);renderer.positions(L,L.cur);
    if(isMask)masks.push({L,side:L.side});
    if(alpha>=0.004)items.push({L,alpha,clip:L.bn==='irides'&&L.side?L.side:null});
  }
  const frame={width:cv.width,height:cv.height,offsetX:-renderBox.left,offsetY:-renderBox.top,background:exportBackground(),masks:maskFailed?[]:masks,items};
  try{renderer.draw(frame);}
  catch(err){
    if(maskFailed)throw err;
    maskFailed=true;status(err.message+' (iris clipping disabled)',true);
    frame.masks=[];renderer.draw(frame);
  }
}
let maskFailed=false,closedOnPreviousFrame=false,blinkCount=0;
let last=performance.now(),simulationTime=last,fpsN=0,fpsT=last;
function tick(now){
  requestAnimationFrame(tick);
  if(KAMYII&&now-last<1000/60-1)return;
  const dt=Math.max(0,Math.min(0.05,(now-last)/1000));last=now;
  if(!OBS_MODE)updateMeter(now);
  if(!layers.length||!A||contextLost)return;
  if(anchorMode||!paused||!lastFrame){simulationTime+=dt*1000;lastFrame=animate(simulationTime,dt);dirty=true;}
  if(dirty||capturePending||(recorder&&recorder.active)){render(lastFrame);dirty=false;}
  if(capturePending){
    capturePending=false;
    cv.toBlob(blob=>{if(blob){download(blob,baseFileName()+'.png');status('PNG exported ('+cv.width+' × '+cv.height+' px'+(prefs.exportBg==='transparent'?', transparent':'')+')');}else status('Could not export the PNG',true);},'image/png');
    invalidate();
  }
  fpsN++;if(now-fpsT>500){$('fps').textContent=Math.round(fpsN*1000/(now-fpsT))+' fps';fpsN=0;fpsT=now;}
}
cv.addEventListener('webglcontextlost',ev=>{ev.preventDefault();contextLost=true;capturePending=false;recorder?.cancel();status('Rendering paused; waiting for the GPU to recover',true);});
cv.addEventListener('webglcontextrestored',()=>{
  try{
    const saved=currentRig?settingsSnapshot():null;layers=[];renderer.init();contextLost=false;
    if(currentRig){applyRig(currentRig);applySettings(saved);}
    status('Rendering recovered');
  }catch(err){contextLost=true;status('Rendering could not recover. Reload the page',true);}
});
window.addEventListener('pagehide',()=>{mic.stop(true);camera.stop(true);liveTicker?.stop();liveTicker=null;sync.close();cancelLoad();recorder?.cancel();});

// Microphone sampling and the OBS relay run on a worker-driven ticker so they
// continue while the editor window is covered by OBS (rAF stops then).
let liveTicker=null;
function updateLiveTicker(){
  const need=!OBS_MODE&&(auto.cam||auto.mic);
  if(need&&!liveTicker)liveTicker=window.RigDevices.createTicker(20,()=>{
    micLevel=auto.mic&&mic.active?mic.sample(prefs.micGain,prefs.micGate):0;
    if(auto.cam||auto.mic)sync.publishTracking(liveTracking());
  });
  else if(!need&&liveTicker){liveTicker.stop();liveTicker=null;micLevel=0;}
}

// Read-only diagnostics for automated tests and bug reports.
Object.defineProperty(window,'Anime25D',{value:Object.freeze({version:APP_VERSION,state(){
  return {modelId,modelName,layers:layers.filter(L=>!L.companion).length,paused,anchorMode,blinkEditor,blinkView,activePreset,background,unsaved,
    anchors:JSON.parse(JSON.stringify(anchorOffsets)),params:Object.assign({},T),auto:Object.assign({},auto),
    pose:lastFrame?{eyeL:lastFrame.eyeOpenL,eyeR:lastFrame.eyeOpenR,mouth:lastFrame.mouthOpen,angleX:lastFrame.angleX,tilt:lastFrame.angleZ,torso:lastFrame.torso?{...lastFrame.torso}:null,followX:lastFrame.followX,followY:lastFrame.followY,tailSway:lastFrame.tailSway,earL:lastFrame.earL,earR:lastFrame.earR,nod:lastFrame.nodPose?.head||0}:null,
    companions:{ears:T.neko>.5,tail:T.tail>.5},
    synth:currentRig?Object.assign({},currentRig.synth):null,
    shading:{vertices:layers.reduce((n,L)=>n+(L.shadingBindings?.length||0),0),pixels:layers.reduce((n,L)=>!L.flow?n:L.flow.reduce((m,v,i)=>Math.max(m,Math.abs(v)*(i%2?L.h:L.w)),n),0)},
    cam:Object.assign({},cam),camLive:cam.live&&performance.now()-lastTrackingAt<1200,calibrated:!!prefs.calibration,
    canvas:{width:cv.width,height:cv.height,offsetX:-renderBox.left,offsetY:-renderBox.top},view:Object.assign({},view),obs:sync.status,recording:!!(recorder&&recorder.active)};
}})});

// ---------- start ----------
if(OBS_MODE){
  auto.cam=true;
  sync.listen();
  (async()=>{
    await sync.ready;
    if(sync.status.relay&&sync.status.serverModel){await loadFromRelay(sync.status.serverModel.id);if(layers.length||relayLoadingId)return;}
    if(layers.length||relayLoadingId)return;   // a model arrived from the editor meanwhile
    const path=RT.safeRelativePath(QUERY.get('model')||'','.psd');
    if(path)await loadModel(fetchSource(path),path.split('/').pop(),{relay:false});
    else await loadSample('B');
  })();
}else{
  if(!KAMYII)sync.startPolling();
  if(KAMYII)loadModel(fetchSource(KAMYII.psd),KAMYII.name,{relay:false,kamyii:true});
  if(CAMERA_MODE)setAuto('cam',true);
}
requestAnimationFrame(tick);
})();
