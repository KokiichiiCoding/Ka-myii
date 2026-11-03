// Animal ears and a tail fitted to a VRM on the 3D stage, like the Live Preview's: they sit on
// the head and hips bones and swing with simple springs. Stage-only: not written into the VRM.
import * as THREE from 'three';

export const EARS = ['off','cat','fox','wolf','bunny','bear'];
export const TAILS = ['off','cat','fox','wolf','devil'];
const FUR = {ivory:[238,226,217],charcoal:[66,65,78],copper:[175,110,76]};
// Outlines in ear widths, base on y=0; inner = colour of the inside of the ear.
const EAR = {
  cat:{h:1,tilt:.28,inner:[206,147,158],points:t=>[[-.5,0],[-.42,.45],[-.12,.92],[0,1],[.12,.92],[.42,.45],[.5,0]]},
  fox:{h:1.35,tilt:.3,inner:[245,237,227],tip:[52,38,42],points:t=>[[-.55,0],[-.45,.55],[-.12,1.25],[0,1.35],[.12,1.25],[.45,.55],[.55,0]]},
  wolf:{h:1.15,tilt:.22,inner:null,points:t=>[[-.46,0],[-.38,.5],[-.1,1.07],[0,1.15],[.1,1.07],[.38,.5],[.46,0]]},
  bunny:{h:2.6,tilt:.1,inner:[234,171,182],points:t=>Array.from({length:21},(_,i)=>{const a=Math.PI*i/20;return [-.3*Math.cos(a),2.6*Math.sin(a)*(i>0&&i<20?1:0)];})},
  bear:{h:.75,tilt:.5,inner:null,points:t=>Array.from({length:21},(_,i)=>{const a=Math.PI*i/20;return [-.5*Math.cos(a),.75*Math.sin(a)];})},
};

// .8: the stage lights brighten these Lambert parts more than the avatar's own toon shading.
const colour = (rgb,d=0) => new THREE.Color(...rgb.map(v=>.8*Math.max(0,Math.min(255,v+d))/255));
function shape(points,scale=1,lift=0){
  const s=new THREE.Shape();
  points.forEach(([x,y],i)=>i ? s.lineTo(x*scale,y*scale+lift) : s.moveTo(x*scale,y*scale+lift));
  return s;
}
function solid(s,depth,rgb){
  const g=new THREE.ExtrudeGeometry(s,{depth,bevelEnabled:true,bevelThickness:depth*.4,bevelSize:depth*.4,bevelSegments:2,curveSegments:6});
  return new THREE.Mesh(g,new THREE.MeshLambertMaterial({color:colour(rgb)}));
}
// The part of an outline above height ``cut`` (a fox's dark ear tip).
function above(points,cut){
  const out=[];
  for(let i=0;i<points.length;i++){
    const [x0,y0]=points[i],[x1,y1]=points[(i+1)%points.length];
    if(y0>=cut)out.push([x0,y0]);
    if((y0-cut)*(y1-cut)<0){const t=(cut-y0)/(y1-y0);out.push([x0+(x1-x0)*t,cut]);}
  }
  return out;
}
function makeEar(kind,width,rgb){
  const spec=EAR[kind],group=new THREE.Group(),points=spec.points(),depth=width*.06;
  group.add(solid(shape(points,width),depth,rgb));
  if(spec.tip)group.add(solid(shape(above(points,spec.h*.68),width),depth*1.1,spec.tip));
  const inner=spec.inner || rgb.map(v=>v+55);
  const front=solid(shape(points.map(([x,y])=>[x*.55,y*.72]),width,width*.05),depth*.5,inner);
  front.position.z=depth*1.2;group.add(front);
  return group;
}

// Tail centre line (u = 0 at the root) in the tail's frame: +Z is the avatar's front. It sweeps
// out to one side, so a front camera sees it beside the body as in the Live Preview.
function tailPoint(u,length,side,sway,phase,kind){
  const curl=kind==='devil' ? .35 : 1;
  const wave=sway*Math.sin(phase-u*2.2)*Math.pow(u,1.5);
  return new THREE.Vector3(length*(side*(.8*u-.1*u*u)+.38*wave),length*(-.25*Math.sin(Math.PI*u*.9)+.55*u*u*curl),-length*(.58*u-.12*u*u));
}
function tailRadius(u,r,kind){
  if(kind==='fox')return r*(.55+1.9*Math.pow(Math.sin(Math.PI*Math.min(1,u*1.05)),.7))*(u>.97 ? .4 : 1);
  if(kind==='wolf')return r*(.6+1.4*Math.pow(Math.sin(Math.PI*Math.min(1,u*1.05)),.8))*(u>.97 ? .4 : 1);
  if(kind==='devil')return r*.42;
  return r*(1-.3*u)*(u>.97 ? .5 : 1);
}
const RINGS=18, SIDES=10;
function makeTail(kind,length,radius,rgb){
  const n=(RINGS+1)*SIDES,positions=new Float32Array(n*3),colours=new Float32Array(n*3),index=[];
  const tipColour=kind==='fox' ? [248,241,232] : kind==='wolf' ? rgb.map(v=>v-70) : null;
  const body=kind==='devil' ? [61,38,57] : rgb;
  for(let i=0;i<=RINGS;i++)for(let k=0;k<SIDES;k++){
    const u=i/RINGS,c=colour(tipColour && u>.8 ? tipColour : body,(k%2 ? -6 : 6));
    colours.set([c.r,c.g,c.b],(i*SIDES+k)*3);
    if(i<RINGS){const a=i*SIDES+k,b=i*SIDES+(k+1)%SIDES;index.push(a,b,a+SIDES,b,b+SIDES,a+SIDES);}
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(positions,3));g.setAttribute('color',new THREE.BufferAttribute(colours,3));g.setIndex(index);
  const mesh=new THREE.Mesh(g,new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.DoubleSide}));
  mesh.frustumCulled=false;   // the vertices move every frame
  const tail={mesh,kind,length,radius,spade:null};
  if(kind==='devil'){
    const w=length*.16,s=shape([[0,0],[.5,.35],[.62,.7],[.4,.95],[0,.7],[-.4,.95],[-.62,.7],[-.5,.35]],w);
    tail.spade=solid(s,w*.12,body);mesh.add(tail.spade);
  }
  return tail;
}
function bendTail(tail,side,sway,phase){
  const {mesh,kind,length,radius}=tail,pos=mesh.geometry.attributes.position;
  const up=new THREE.Vector3(),right=new THREE.Vector3(),t=new THREE.Vector3();
  for(let i=0;i<=RINGS;i++){
    const u=i/RINGS,p=tailPoint(u,length,side,sway,phase,kind);
    t.subVectors(tailPoint(Math.min(1,u+.02),length,side,sway,phase,kind),tailPoint(Math.max(0,u-.02),length,side,sway,phase,kind)).normalize();
    right.crossVectors(t,Math.abs(t.y)>.9 ? new THREE.Vector3(1,0,0) : new THREE.Vector3(0,1,0)).normalize();up.crossVectors(right,t);
    const r=tailRadius(u,radius,kind);
    for(let k=0;k<SIDES;k++){
      const a=2*Math.PI*k/SIDES;
      pos.setXYZ(i*SIDES+k,p.x+r*(Math.cos(a)*right.x+Math.sin(a)*up.x),p.y+r*(Math.cos(a)*right.y+Math.sin(a)*up.y),p.z+r*(Math.cos(a)*right.z+Math.sin(a)*up.z));
    }
    if(i===RINGS){tail.tip=p;if(tail.spade){tail.spade.position.copy(p);tail.spade.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),t);}}
  }
  pos.needsUpdate=true;mesh.geometry.computeVertexNormals();
}

// A damped spring toward ``target``, in small steps so a slow frame cannot blow it up.
function spring(s,target,k,c,dt){
  for(let n=Math.ceil(dt/.016),h=dt/Math.max(1,n),i=0;i<n;i++){s.v+=((target-s.x)*k-s.v*c)*h;s.x+=s.v*h;}
}
export function createCompanions(){
  let fit=null,ears=[],tail=null,settings={ear:'off',tail:'off',fur:'hair',earSize:1,earLift:0,tailSize:1};
  const springs={earL:{x:0,v:0},earR:{x:0,v:0},tail:{x:0,v:0}},last={head:null,hips:null};
  let twitchAt=performance.now()+3000,twitchSide=0,twitchStart=-1;
  function clear(){
    for(const e of ears){e.parent?.remove(e);e.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});}
    if(tail){tail.group.parent?.remove(tail.group);tail.mesh.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});}
    ears=[];tail=null;
  }
  // Measure the avatar once: head top and width, the back of the hips, the hair colour.
  function measure(vrm){
    const bone=n=>vrm.humanoid?.getRawBoneNode(n),head=bone('head'),hips=bone('hips');
    if(!head || !hips)return null;
    vrm.scene.updateMatrixWorld(true);
    const meshes=[];vrm.scene.traverse(o=>{if(o.isMesh)meshes.push(o);});
    const ray=new THREE.Raycaster(),H=head.getWorldPosition(new THREE.Vector3()),P=hips.getWorldPosition(new THREE.Vector3());
    const cast=(from,dir)=>{ray.set(from,dir.normalize());return ray.intersectObjects(meshes,false)[0] || null;};
    const box=new THREE.Box3().setFromObject(vrm.scene),height=box.max.y-box.min.y;
    // Head width halfway up the head (its pivot sits near the chin on Ka-myii rigs).
    const top=cast(new THREE.Vector3(H.x,box.max.y+1,H.z),new THREE.Vector3(0,-1,0)),headH=top ? Math.max(top.point.y-H.y,height*.05) : height*.13;
    const y=H.y+headH*.55,l=cast(new THREE.Vector3(H.x-5,y,H.z),new THREE.Vector3(1,0,0)),r=cast(new THREE.Vector3(H.x+5,y,H.z),new THREE.Vector3(-1,0,0));
    const width=l && r ? Math.min(r.point.x-l.point.x,headH*1.6) : headH;
    const tops=[-1,1].map(s=>cast(new THREE.Vector3(H.x+s*width*.3,box.max.y+1,H.z),new THREE.Vector3(0,-1,0)));
    if(tops.some(t=>!t))return null;
    let back=null;   // the back at the hips, a little higher if there is a gap there; a bust has no hips to hang a tail on
    if(box.min.y<P.y-height*.12)for(const dy of [-.02,0,.03,.06])back=back || cast(new THREE.Vector3(P.x,P.y+dy*height,P.z-5),new THREE.Vector3(0,0,1));
    // Where to read the hair colour off the rendered stage: the hair beside the face, seen from the
    // front (a hat makes the crown useless), sized by the shoulders; the median skips highlights.
    const la=bone('leftUpperArm'),ra=bone('rightUpperArm');
    const shoulders=la && ra ? la.getWorldPosition(new THREE.Vector3()).distanceTo(ra.getWorldPosition(new THREE.Vector3())) : width*1.6;
    const probes=[];
    const probe=(dx,dy)=>{
      const hit=cast(new THREE.Vector3(H.x+dx*shoulders,H.y+dy*shoulders,H.z+5),new THREE.Vector3(0,0,-1));
      if(hit)probes.push(head.worldToLocal(hit.point.clone().add(new THREE.Vector3(0,0,height*.002))));
    };
    for(const dy of [.3,.42,.54])for(const dx of [-.4,-.34,.34,.4])probe(dx,dy);
    if(probes.length<4)for(const dx of [-.1,0,.1])probe(dx,headH/shoulders*.85);   // a narrow head: the bangs
    return {head,hips,width,height,tops,back,probes,hair:null};
  }
  function attach(node,object,world){   // keep the world pose of the moment, as a child of the bone
    node.add(object);object.position.copy(node.worldToLocal(world.clone()));
    object.quaternion.copy(node.getWorldQuaternion(new THREE.Quaternion()).invert());
  }
  function build(){
    clear();
    if(!fit)return;
    const hair=fit.hair || [190,177,199],rgb=settings.fur==='hair' ? hair : FUR[settings.fur] || hair;
    if(settings.ear!=='off' && EAR[settings.ear]){
      const spec=EAR[settings.ear],w=fit.width*.36*settings.earSize;
      fit.tops.forEach((hit,i)=>{
        const side=i ? 1 : -1,pivot=new THREE.Group(),ear=makeEar(settings.ear,w,rgb);
        ear.position.z=-w*.05;pivot.add(ear);
        const base=hit.point.clone().add(new THREE.Vector3(0,-w*.18+settings.earLift*w*spec.h,0));
        attach(fit.head,pivot,base);
        pivot.userData={side,tilt:spec.tilt,rest:pivot.quaternion.clone()};ears.push(pivot);
      });
    }
    if(settings.tail!=='off' && fit.back){
      const group=new THREE.Group(),t=makeTail(settings.tail,fit.height*.42*settings.tailSize,fit.height*.018*settings.tailSize,rgb);
      group.add(t.mesh);attach(fit.hips,group,fit.back.point.clone().add(new THREE.Vector3(0,0,-fit.height*.01)));
      tail={...t,group};bendTail(tail,1,0,0);
    }
  }
  return {
    EARS,TAILS,
    load(vrm,saved){clear();fit=vrm ? measure(vrm) : null;Object.assign(settings,saved || {});build();return {canTail:!!fit?.back,canEars:!!fit};},
    set(change){Object.assign(settings,change);build();},
    get settings(){return {...settings};},
    // The stage reads ``pixel(worldPoint)`` -> [r,g,b,a] off its canvas after a frame, once.
    wantsColour(){return !!fit && !fit.hair && settings.fur==='hair' && (ears.length>0 || !!tail);},
    colourFrom(pixel){
      const seen=fit.probes.map(l=>pixel(fit.head.localToWorld(l.clone()))).filter(c=>c && c[3]>200);
      fit.hair=seen.length ? [0,1,2].map(c=>seen.map(v=>v[c]).sort((a,b)=>a-b)[seen.length>>1]) : [190,177,199];
      build();
    },
    update(dt,now,motionOn){
      if(!fit)return;
      const headQ=fit.head.getWorldQuaternion(new THREE.Quaternion()),hipsP=fit.hips.getWorldPosition(new THREE.Vector3());
      // Ears lag head turns and flick now and then; the tail swings with the hips.
      let turn=0,move=0;
      if(last.head && dt>0){turn=2*Math.asin(Math.min(1,new THREE.Quaternion().copy(last.head).invert().multiply(headQ).y))/dt;move=(hipsP.x-last.hips.x)/dt/fit.height;}
      last.head=headQ;last.hips=hipsP;
      if(motionOn && now>twitchAt){twitchSide=Math.random()<.5 ? 0 : 1;twitchStart=now;twitchAt=now+2500+Math.random()*5000;}
      const flick=twitchStart>0 ? Math.sin(Math.min(1,(now-twitchStart)/260)*Math.PI)*.35 : 0;
      for(const name of ['earL','earR'])spring(springs[name],Math.max(-.4,Math.min(.4,-turn*.08)),140,9,dt);
      for(const ear of ears){
        const {side,tilt,rest}=ear.userData,i=side<0 ? 0 : 1,lag=springs[i ? 'earR' : 'earL'].x,f=twitchSide===i ? flick : 0;
        ear.quaternion.copy(rest).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(-.12-f*.6,0,-side*(tilt+f*.5)+lag)));
      }
      if(tail){
        const s=springs.tail;spring(s,Math.max(-1,Math.min(1,-move*1.5)),30,5,dt);
        const kind=tail.kind,speed={fox:1.25,wolf:1.1,devil:2.6}[kind] || 1.8,t=now/1000;
        const idle=motionOn ? .5+.25*Math.sin(t*.31)+.6*Math.pow(Math.max(0,Math.sin(t*.53+1)),14) : 0;
        tail.sway=Math.max(-1.8,Math.min(1.8,idle+s.x*3));bendTail(tail,1,tail.sway,t*speed);
      }
    },
    state(){
      const tip=tail?.tip ? tail.mesh.localToWorld(tail.tip.clone()) : null;
      return {ear:ears.length ? settings.ear : 'off',tail:tail ? settings.tail : 'off',ears:ears.map(e=>e.getWorldPosition(new THREE.Vector3()).toArray()),
        earAngles:ears.map(e=>e.quaternion.angleTo(e.userData.rest)),hair:fit?.hair || null,tailSway:tail?.sway || 0,tailTip:tip?.toArray() || null,canTail:!!fit?.back};
    },
  };
}
