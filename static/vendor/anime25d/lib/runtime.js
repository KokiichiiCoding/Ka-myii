/* Shared, browser-independent validation and simulation helpers. */
(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.RigRuntime = factory();
})(typeof self !== 'undefined' ? self : this, function() {
  'use strict';
  const MAX_FILE_BYTES = 128 * 1024 * 1024;
  const SETTINGS_FORMAT = 'anime25d-settings';
  const SETTINGS_VERSION = 2;
  const PRESETS = ['neutral','smile','usume','surprise','jito','winkL','winkR','grin','jawdrop'];
  const AUTO_KEYS = ['idle','blink','rand','talk','mouse','phys'];
  const BACKGROUNDS = ['checker','green','dark'];
  // Editable anchor offsets (PSD pixels) and the axes each one may move.
  const ANCHOR_KEYS = {face:['dx','dy'],eyeL:['dx','dy'],eyeR:['dx','dy'],eyeLClose:['dx','dy'],eyeRClose:['dx','dy'],mouth:['dx','dy'],neck:['dx','dy'],chest:['dx','dy']};
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const isNum = v => typeof v === 'number' && Number.isFinite(v);

  function validateHeader(buffer) {
    if (!buffer || buffer.byteLength < 26) throw new Error('The PSD file is too short');
    if (buffer.byteLength > MAX_FILE_BYTES) throw new Error('The PSD must be 128 MB or smaller');
    const v = new DataView(buffer);
    if (v.getUint32(0) !== 0x38425053 || v.getUint16(4) !== 1) throw new Error('Not a supported PSD file (PSB is not supported)');
    const h = v.getUint32(14), w = v.getUint32(18);
    if (w < 2 || h < 2 || w*h > 24000000 || Math.max(w,h) > 16384) throw new Error('The PSD must be at most 16384 px per side and 24 megapixels');
    if (v.getUint16(22) !== 8 || v.getUint16(24) !== 3) throw new Error('Save the PSD again as RGB, 8 bits per channel');
    return {w,h};
  }
  function fingerprint(buffer) {
    const b = new Uint8Array(buffer); let a = 2166136261, c = 5381;
    for (let i=0; i<b.length; i++) { a = Math.imul(a ^ b[i], 16777619); c = Math.imul(c, 33) ^ b[i]; }
    return b.length.toString(16)+'-'+(a>>>0).toString(16)+'-'+(c>>>0).toString(16);
  }
  function meshSize(w,h,cell) {
    let nx=Math.max(2,Math.round(w/cell)), ny=Math.max(2,Math.round(h/cell));
    while ((nx+1)*(ny+1)>65000) {
      if(nx>=ny) nx=Math.max(2,Math.floor(nx*0.9)); else ny=Math.max(2,Math.floor(ny*0.9));
    }
    return {nx,ny};
  }
  // Semi-implicit Euler with fixed 120 Hz sub-steps: stable from 10 to 240 fps.
  function spring(s,target,k,damping,dt) {
    const count=Math.max(1,Math.ceil(dt/(1/120))), h=dt/count;
    for(let i=0;i<count;i++){ s.v+=(-k*(s.x-target)-damping*s.v)*h; s.x+=s.v*h; }
  }

  // One displacement field is shared by the bangs and the shading baked into skin.
  function strandBinding(strands,x,y) {
    const n=strands.length;
    let spacing=120;
    if(n>1){const d=strands.slice(1).map((s,i)=>s.x-strands[i].x).sort((a,b)=>a-b);spacing=d[d.length>>1];}
    const sig=Math.max(1,spacing*0.6),weights=strands.map(s=>Math.exp(-Math.pow((x-s.x)/sig,2)));
    let total=weights.reduce((a,b)=>a+b,0),root=0,tip=0;
    if(total<1e-6){weights.fill(0);weights[0]=1;total=1;}
    for(let i=0;i<n;i++){weights[i]/=total;root+=weights[i]*strands[i].rootY;tip+=weights[i]*strands[i].tipY;}
    return {weights,u:clamp((y-root)/Math.max(1,tip-root),0,1)};
  }
  function strandShift(binding,springs,amp,soft,front,limit,power) {
    const u=front?Math.min(1,binding.u*1.6):binding.u;
    const mix=Math.min(1,Math.pow(u,1.2)*soft);
    let dx=0;
    for(let i=0;i<springs.length;i++)dx+=binding.weights[i]*(springs[i].stiff.dx*(1-mix)+springs[i].soft.dx*mix);
    dx*=Math.pow(u,power ?? (front?1.8:2.1))*amp;
    // Smooth saturation prevents a fast turn or a long frame from tearing bangs off the face.
    return limit>0?limit*Math.tanh(dx/limit):dx;
  }
  function closedEyePoint(x,y,eye,sourceY,scale,dy,angle) {
    const sourceX=(eye.x0+eye.x1)/2,cx=eye.closeX ?? sourceX,cy=eye.closeY;
    const rx=(x-sourceX)*scale,ry=(y-sourceY)*scale;
    const c=Math.cos(angle),s=Math.sin(angle);
    return [cx+rx*c-ry*s,cy+dy+rx*s+ry*c];
  }

  // ---- face tracking relay ----
  const trackingKeys={ax:[-1,1],ay:[-1,1],az:[-1,1],eL:[0,1],eR:[0,1],mo:[0,1],ex:[-1,1],ey:[-1,1]};
  const optionalTrackingKeys={br:[-1,1],mf:[-1,1],mic:[0,1]};
  function tracking(value) {
    if(!value || typeof value!=='object' || typeof value.live!=='boolean') return null;
    const out={live:value.live};
    for(const [key,range] of Object.entries(trackingKeys)) {
      if(!isNum(value[key]))return null;
      out[key]=clamp(value[key],...range);
    }
    for(const [key,range] of Object.entries(optionalTrackingKeys)) {
      if(value[key]===undefined) continue;
      if(!isNum(value[key]))return null;
      out[key]=clamp(value[key],...range);
    }
    return out;
  }

  // ---- per-model settings ----
  function anchors(value, limit) {
    const out={};
    if(value==null) return out;
    if(typeof value!=='object' || Array.isArray(value)) throw new Error('Invalid anchor settings');
    const lim=isNum(limit)&&limit>0?limit:16384;
    for(const [key,axes] of Object.entries(ANCHOR_KEYS)) {
      const rec=value[key];
      if(rec==null) continue;
      if(typeof rec!=='object') throw new Error('Invalid anchor settings: '+key);
      const clean={};
      for(const axis of axes) {
        if(rec[axis]===undefined) continue;
        if(!isNum(rec[axis])) throw new Error('Invalid anchor settings: '+key);
        const n=clamp(rec[axis],-lim,lim);
        if(n!==0) clean[axis]=Math.round(n*100)/100;
      }
      if(Object.keys(clean).length) out[key]=clean;
    }
    return out;
  }
  function settings(value,modelId,ranges,layerIds,defaults,options) {
    const opts=options||{};
    if (!value || value.format!==SETTINGS_FORMAT || !(value.version===1 || value.version===2)) throw new Error('Not a supported settings file');
    if (!opts.anyModel && value.modelId!==modelId) throw new Error('These settings belong to another PSD. Load the PSD they were saved with');
    if(!value.params || typeof value.params!=='object' || !Array.isArray(value.layers)) throw new Error('Invalid settings structure');
    const out={params:{},auto:{},layers:[],background:'checker',preset:null,anchors:{},missingLayers:0,unknownLayers:0};
    if(PRESETS.includes(value.preset))out.preset=value.preset;
    for(const [key,range] of Object.entries(ranges)) {
      const n=value.params[key];
      if(n===undefined){ if(defaults&&isNum(defaults[key]))out.params[key]=clamp(defaults[key],...range); continue; }
      if(!isNum(n)) throw new Error('Invalid number setting: '+key);
      out.params[key]=clamp(n,...range);
    }
    for(const key of AUTO_KEYS) {
      const v=value.auto?.[key];
      if(v===undefined) continue;
      if(typeof v!=='boolean') throw new Error('Invalid animation settings');
      out.auto[key]=v;
    }
    const known=new Set(layerIds), seen=new Set();
    for(const l of value.layers) {
      if(!l || typeof l.id!=='string' || seen.has(l.id) || typeof l.visible!=='boolean' ||
        !isNum(l.opacity) || !isNum(l.depth)) throw new Error('Invalid layer settings');
      seen.add(l.id);
      if(!known.has(l.id)){out.unknownLayers++;continue;}
      out.layers.push({id:l.id,visible:l.visible,opacity:clamp(l.opacity,0,1),depth:clamp(l.depth,0,2)});
    }
    out.missingLayers=layerIds.filter(id=>!seen.has(id)).length;
    if(BACKGROUNDS.includes(value.background))out.background=value.background;
    out.anchors=anchors(value.anchors,opts.anchorLimit);
    return out;
  }
  // Merge a saved layer order into the current layer list. Layers the saved
  // file does not mention keep their default neighbours.
  function mergeLayerOrder(defaultIds, savedIds) {
    const saved=savedIds.filter(id=>defaultIds.includes(id));
    const out=saved.slice();
    defaultIds.forEach((id,i)=>{
      if(out.includes(id)) return;
      let at=0;
      for(let j=i-1;j>=0;j--){const k=out.indexOf(defaultIds[j]); if(k>=0){at=k+1;break;}}
      out.splice(at,0,id);
    });
    return out;
  }

  // ---- global preferences (per browser, not per model) ----
  // schema: {key: [min,max,default] | ['bool',default] | ['enum',[...],default]}
  function prefs(value, schema) {
    const out={};
    const src=value&&typeof value==='object'?value:{};
    for(const [key,spec] of Object.entries(schema)) {
      const v=src[key];
      if(spec[0]==='bool') out[key]=typeof v==='boolean'?v:spec[1];
      else if(spec[0]==='enum') out[key]=spec[1].includes(v)?v:spec[2];
      else if(spec[0]==='any') out[key]=v===undefined?spec[1]:v;
      else out[key]=isNum(v)?clamp(v,spec[0],spec[1]):spec[2];
    }
    return out;
  }

  // ---- undo / redo over serialized snapshots ----
  function createHistory(limit) {
    const max=Math.max(2,limit||100);
    let stack=[], index=-1;
    return {
      reset(snap){stack=[snap];index=0;},
      push(snap){
        if(index>=0&&stack[index]===snap)return false;
        stack=stack.slice(0,index+1);stack.push(snap);
        if(stack.length>max)stack.shift();
        index=stack.length-1;return true;
      },
      undo(){return index>0?stack[--index]:null;},
      redo(){return index<stack.length-1?stack[++index]:null;},
      current(){return index>=0?stack[index]:null;},
      get canUndo(){return index>0;},
      get canRedo(){return index<stack.length-1;},
      get size(){return stack.length;}
    };
  }

  // ---- misc ----
  // Relative same-origin file reference for ?model= / ?settings= parameters.
  function safeRelativePath(p, ext) {
    if(typeof p!=='string' || !p || p.length>260) return null;
    if(/[\u0000-\u001f\\:?#%]/.test(p) || p.startsWith('/') || p.split('/').some(seg=>seg==='..'||seg==='.'||seg==='')) return null;
    if(ext && !p.toLowerCase().endsWith(ext)) return null;
    return p;
  }
  function safeFileName(name, fallback) {
    const base=String(name||'').replace(/\.[a-z0-9]{1,5}$/i,'').replace(/[\u0000-\u001f<>:"/\\|?*]+/g,'_').trim().slice(0,80);
    return base||fallback||'model';
  }
  function formatTime(seconds) {
    const s=Math.max(0,seconds);
    return (s<10?s.toFixed(1):Math.round(s))+' s';
  }

  return {MAX_FILE_BYTES,SETTINGS_FORMAT,SETTINGS_VERSION,PRESETS,AUTO_KEYS,BACKGROUNDS,ANCHOR_KEYS,
    validateHeader,fingerprint,meshSize,spring,strandBinding,strandShift,closedEyePoint,tracking,settings,anchors,mergeLayerOrder,prefs,createHistory,
    safeRelativePath,safeFileName,formatTime,clamp};
});
