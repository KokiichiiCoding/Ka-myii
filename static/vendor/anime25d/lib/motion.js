/* Ka-myii preview motion and optional illustrated companions. MIT. */
(function(root,factory){
  if(typeof module==='object'&&module.exports)module.exports=factory();
  else root.RigMotion=factory();
})(typeof self!=='undefined'?self:this,function(){
  'use strict';
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const smooth=v=>{v=clamp(v,0,1);return v*v*(3-2*v);};
  // Blend while the open drawing is still compressed, rather than popping to a half-open mouth.
  function mouth(open,form,round,ease){
    open=clamp(open,0,1);form=clamp(form,-1,1);round=clamp(round,0,1);
    return {alpha:smooth((open-.015)/(.18+.10*ease)),
      height:.08+.92*Math.pow(open,.85),width:1+.12*form-.28*round*open};
  }
  function response(value,target,dt,speed){
    const rate=target>value?20+speed*24:12+speed*16;
    return value+(target-value)*(1-Math.exp(-Math.max(0,dt)*rate));
  }
  // A head-led dip/rebound: shoulders travel much less and arrive slightly later.
  // Sway tilts side to side at half the nod tempo and lingers at each side.
  function nodPose(seconds,depth,hz,sway){
    const phase=seconds*Math.PI*2*hz,wave=Math.sin(phase)+.14*Math.sin(phase*2);
    return {head:.10*wave*depth,body:.015*Math.sin(phase-.5)*depth,
      squash:1-.035*Math.max(0,wave)*depth,pitch:-.12*wave*depth,
      tilt:sway?sway*Math.tanh(2.2*Math.sin(phase/2))/Math.tanh(2.2):0};
  }
  function nodPoint(y,pivot,faceHeight,pose,weight){
    return y+faceHeight*(pose.head*weight+pose.body)+(y-pivot)*(pose.squash-1)*weight;
  }
  // Head roll in radians at |angleZ| = 1 (about 16°).
  const ROLL=.28;
  // Roll about the neck pivot. Below the chin a point slides sideways like the chin line above
  // it, so hanging hair and the neck follow the head instead of swinging the opposite way; its
  // lift fades out over `fall` px so hair cut off by the canvas edge never rises off it.
  function rollPoint(x,y,pivot,chinY,angle,weight,fall){
    const rx=x-pivot.cx,ry=Math.min(y,chinY)-pivot.cy,c=Math.cos(angle),s=Math.sin(angle);
    const lift=y>chinY?1-smooth((y-chinY)/(fall||Infinity)):1;
    return [x+(rx*c-ry*s-rx)*weight,y+(rx*s+ry*c-ry)*weight*lift];
  }
  // Opaque left/right extents per band of rows, in canvas coordinates: a tight outline for bounds.
  function rowExtents(img,left,top,bands){
    const d=img.data,w=img.width,h=img.height,step=Math.max(1,Math.ceil(h/bands)),pts=[];
    for(let y0=0;y0<h;y0+=step){
      const y1=Math.min(h,y0+step);let a=w,b=-1;
      for(let y=y0;y<y1;y++)for(let x=0;x<w;x++)if(d[(y*w+x)*4+3]>8){if(x<a)a=x;if(x>b)b=x;}
      if(b>=0)pts.push([left+a,top+y0],[left+b+1,top+y0],[left+a,top+y1],[left+b+1,top+y1]);
    }
    return pts;
  }
  // How much a body point follows the head: fully at the chin, `base` at the Neck pivot, fading
  // out `drop` px below it. A collar within `half` of the neck's centre line moves the same way
  // as the neck under it; the follow fades out over another `half` towards the shoulders.
  function collarWeight(x,y,pivot,chinY,half,drop,base){
    const v=y<pivot.cy?base+(1-base)*smooth((pivot.cy-y)/Math.max(1,pivot.cy-chinY)):base*(1-smooth((y-pivot.cy)/drop));
    return v*(1-smooth((Math.abs(x-pivot.cx)-half)/half));
  }
  // Upward shift of a shoulder point: none within `inner` px of the centre line, all of it past
  // `outer`, fading out `drop` px below the shoulder line (a cropped chest stays on the canvas
  // bottom). Arms (`rigid`) rise whole with their shoulder.
  function shoulderLift(x,y,pivot,inner,outer,drop,left,right,rigid){
    const wx=smooth((Math.abs(x-pivot.cx)-inner)/(outer-inner));
    return (x<pivot.cx?left:right)*wx*(rigid||y<=pivot.cy?1:1-smooth((y-pivot.cy)/drop));
  }
  // A torso turn: the centre line moves further than the outline; legs below the hips stay put.
  function turnShift(x,y,cx,half,hipY,fall,amount){
    return amount*(1-.7*smooth(Math.abs(x-cx)/half))*(1-smooth((y-hipY)/fall));
  }
  // Shared torso parent: an arc above the hips, fading to fixed feet/cropped lower edges.
  // All head parts inherit the same chest transform; lower hair and clothing blend down.
  function torsoPoint(x,y,sourceY,g,pose){
    const w=1-smooth((sourceY-g.shoulderY)/Math.max(1,g.hipY-g.shoulderY));
    const shift=clamp(pose.shift||0,-2,2),root=1-smooth((sourceY-g.hipY)/Math.max(1,(g.footY||g.hipY)-g.hipY));
    if(w<=0&&(!shift||root<=0))return [x,y];
    const angle=clamp(pose.roll,-1,1)*.10*w,c=Math.cos(angle),sn=Math.sin(angle);
    const rx=x-g.cx,ry=y-g.hipY;
    const turn=clamp(pose.yaw,-1,1)*g.width*.075*w;
    const front=1-.22*smooth(Math.abs(rx)/Math.max(1,g.width*.55));
    return [g.cx+rx*c-ry*sn+turn*front+shift*g.width*.09*root,
      g.hipY+rx*sn+ry*c+pose.lift*g.height*.018*w+Math.abs(shift)*g.height*.006*root];
  }
  // Deliberate weight transfers with a quiet dwell at each end, rather than a constant
  // metronome sway. Quintic interpolation has zero speed/acceleration at the joins.
  function idlePose(seconds){
    const poses=[-.45,.45,.18,-.32],phase=((seconds%16)+16)%16/4,i=Math.floor(phase),u=phase-i;
    const v=clamp((u-.18)/.64,0,1),q=v*v*v*(v*(v*6-15)+10);
    const shift=poses[i]+(poses[(i+1)%poses.length]-poses[i])*q;
    return {shift,roll:-shift*.18};
  }
  // Live2D's sample app idles the head with incommensurate periods (6.5345/3.5345/5.5345 s), so
  // the loop never visibly repeats; amplitudes here are ~60% of its ±7.5°/±4°/±5° (2.5D art).
  function idleHead(seconds){
    const w=p=>Math.sin(seconds*Math.PI*2/p);
    return {x:.15*w(6.5345),y:.08*w(3.5345),z:.10*w(5.5345)};
  }
  // Live2D's sample motions tilt the head with the turn (Angle Z follows Angle X, r≈0.9).
  const TURN_TILT=.3;
  // Eye openness `d` s into a blink, or -1 when done. The samples close in ~0.1 s and open in
  // ~0.2 s: lids accelerate shut and ease open. `hold` keeps them closed (a slow blink).
  function blink(d,hold){
    if(d<.09){const u=d/.09;return 1-u*u;}
    if(d<.09+hold)return 0;
    const u=(d-.09-hold)/.18;
    return u<1?1-Math.pow(1-u,3):-1;
  }
  function breath(seconds){
    const p=(((seconds%4.8)+4.8)%4.8)/4.8,v=p<.4?p/.4:1-(p-.4)/.6;
    return smooth(v);
  }
  // A hat, bow or ribbon trails the head by more the further it reaches from the crown.
  function hatWeight(x,y,crown,radius){return smooth((Math.hypot(x-crown.cx,y-crown.cy)-radius)/radius);}
  function rotate(x,y,a){const c=Math.cos(a),s=Math.sin(a);return [x*c-y*s,x*s+y*c];}
  // A curved ribbon follows a continuous centreline: its root remains fixed for every pose.
  function tailPoint(x,u,length,width,side,phase,sway){
    const bend=sway*Math.sin(phase-u*2.1)*u*u;
    const px=side*length*(.72*u+.22*Math.sin(Math.PI*u)+.16*bend);
    const py=-length*(.12*u+.66*u*u)+length*.10*sway*Math.sin(phase-u*2.1)*u*u;
    const tx=side*(.72+.22*Math.PI*Math.cos(Math.PI*u));
    const ty=-.12-1.32*u,n=Math.hypot(tx,ty);
    return [px-ty/n*x*width,py+tx/n*x*width];
  }
  function earPoint(x,y,width,height,side,angle){return rotate(x*width,y*height,side*angle);}
  function palette(rig,index){
    const fixed=[null,[238,226,217],[66,65,78],[175,110,76]];
    if(index>0)return fixed[index]||fixed[1];
    const samples=[[],[],[]];
    for(const L of rig.layers){
      if(!/^(front hair|back hair|side hair)(_|$)/.test(L.name))continue;
      const d=L.img.data;
      for(let i=0;i<d.length;i+=4*17){
        if(d[i+3]<240||Math.max(d[i],d[i+1],d[i+2])<45)continue;
        for(let c=0;c<3;c++)samples[c].push(d[i+c]);
      }
    }
    return samples.map((v,c)=>v.length?v.sort((a,b)=>a-b)[v.length>>1]:[190,177,199][c]);
  }
  function art(kind,rgb){
    const canvas=document.createElement('canvas');canvas.width=128;canvas.height=kind==='tail'?384:160;
    const c=canvas.getContext('2d'),colour=(amount)=>'rgb('+rgb.map(v=>Math.round(clamp(v+amount,0,255))).join(',')+')';
    c.lineJoin='round';c.lineCap='round';
    if(kind==='tail'){
      const g=c.createLinearGradient(32,0,96,0);g.addColorStop(0,colour(-38));g.addColorStop(.45,colour(8));g.addColorStop(1,colour(-20));
      c.beginPath();c.moveTo(38,380);c.bezierCurveTo(32,280,32,100,43,34);c.quadraticCurveTo(64,-13,85,34);c.bezierCurveTo(96,100,96,280,90,380);c.closePath();
      c.fillStyle=g;c.fill();c.strokeStyle=colour(-75);c.lineWidth=3;c.stroke();
      c.save();c.clip();c.fillStyle=colour(30);c.beginPath();c.ellipse(64,38,35,36,0,0,Math.PI*2);c.fill();c.restore();
      c.beginPath();c.moveTo(48,320);c.quadraticCurveTo(41,190,53,85);c.strokeStyle=colour(24);c.lineWidth=3;c.stroke();
    }else{
      const g=c.createLinearGradient(20,30,108,155);g.addColorStop(0,colour(22));g.addColorStop(1,colour(-35));
      c.beginPath();c.moveTo(9,151);c.bezierCurveTo(11,109,24,42,52,9);c.quadraticCurveTo(61,-2,69,11);c.bezierCurveTo(100,49,116,106,119,151);c.quadraticCurveTo(101,140,93,151);c.lineTo(83,142);c.lineTo(74,153);c.lineTo(63,145);c.lineTo(51,155);c.lineTo(39,145);c.quadraticCurveTo(21,143,9,151);c.closePath();
      c.fillStyle=g;c.fill();c.strokeStyle=colour(-75);c.lineWidth=3;c.stroke();
      c.beginPath();c.moveTo(30,128);c.quadraticCurveTo(39,62,62,31);c.quadraticCurveTo(85,65,99,128);c.quadraticCurveTo(62,105,30,128);
      c.fillStyle='#ce939e';c.fill();c.strokeStyle='#a66f83';c.lineWidth=2;c.stroke();
      c.beginPath();c.moveTo(40,118);c.quadraticCurveTo(48,77,61,51);c.strokeStyle='#e8b3bd';c.lineWidth=4;c.stroke();
      c.beginPath();c.moveTo(22,131);c.lineTo(37,119);c.moveTo(101,131);c.lineTo(89,118);c.strokeStyle=colour(28);c.lineWidth=4;c.stroke();
    }
    return c.getImageData(0,0,canvas.width,canvas.height);
  }
  function mouthWeights(open,teeth,round,enabled){
    const gate=smooth((open-.025)/.18)*clamp(enabled,0,1);
    const o=clamp(round,0,1)*gate,grin=clamp(teeth,0,1)*gate*(1-o);
    return {round:o,grin,source:1-o-grin};
  }
  // Estimate visible mouth ink rather than the centre/width of an inpainted skin patch.
  function mouthBox(rig){
    const a=rig.anchors.mouth,fw=rig.anchors.face.x1-rig.anchors.face.x0;
    const candidates=rig.layers.filter(L=>/^mouth_(close|open)$/.test(L.name)&&!L.synthetic).sort((a,b)=>(a.name==='mouth_close'?0:1)-(b.name==='mouth_close'?0:1));
    for(const L of candidates){
      let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity,count=0;
      for(let y=0;y<L.h;y++)for(let x=0;x<L.w;x++){
        const i=(y*L.w+x)*4,d=L.img.data;
        if(d[i+3]<150||d[i]*.2126+d[i+1]*.7152+d[i+2]*.0722>125)continue;
        x0=Math.min(x0,L.x+x);x1=Math.max(x1,L.x+x);y0=Math.min(y0,L.y+y);y1=Math.max(y1,L.y+y);count++;
      }
      if(count>=3&&x1-x0<fw*.3)return {cx:(x0+x1)/2,cy:(y0+y1)/2,w:clamp(x1-x0+1,fw*.065,fw*.22)};
    }
    return {cx:a.cx,cy:a.cy,w:clamp(a.x1-a.x0,fw*.065,fw*.20)};
  }
  function mouthArt(kind){
    const canvas=document.createElement('canvas');canvas.width=192;canvas.height=160;
    const c=canvas.getContext('2d');c.lineJoin='round';c.lineCap='round';
    const outline=new Path2D();
    if(kind==='round')outline.ellipse(96,80,39,60,0,0,Math.PI*2);
    else{outline.moveTo(21,44);outline.quadraticCurveTo(96,30,171,44);outline.bezierCurveTo(155,128,42,133,21,44);outline.closePath();}
    const inner=c.createLinearGradient(0,22,0,143);inner.addColorStop(0,'#402b37');inner.addColorStop(.65,'#783e4c');inner.addColorStop(1,'#ab6070');
    c.fillStyle=inner;c.fill(outline);c.save();c.clip(outline);
    if(kind==='grin'){
      const teeth=c.createLinearGradient(0,38,0,78);teeth.addColorStop(0,'#ddd2cc');teeth.addColorStop(.28,'#fff9ee');teeth.addColorStop(1,'#eee1d8');
      c.fillStyle=teeth;c.beginPath();c.moveTo(10,31);c.lineTo(182,31);c.lineTo(166,67);c.quadraticCurveTo(96,83,26,67);c.closePath();c.fill();
      c.strokeStyle='#bd8e91';c.lineWidth=1.2;c.beginPath();c.moveTo(28,68);c.quadraticCurveTo(96,82,165,68);c.stroke();
    }
    c.fillStyle='#d3818c';c.beginPath();c.ellipse(96,kind==='round'?137:124,kind==='round'?34:55,23,0,0,Math.PI*2);c.fill();
    c.restore();c.strokeStyle='#865264';c.lineWidth=3;c.stroke(outline);
    return c.getImageData(0,0,canvas.width,canvas.height);
  }
  function mouthParts(rig){
    const box=mouthBox(rig),existing=rig.layers.filter(L=>/^mouth_(open|close)$/.test(L.name));
    if(!existing.length)return [];
    const z=Math.max(...existing.map(L=>L.z)),w=box.w*1.3,h=box.w*1.08;
    return ['grin','round'].map((expression,i)=>({name:'mouth_'+expression,expression,source:expression==='grin'?'Preview toothy smile':'Preview O mouth',
      x:box.cx-w/2,y:box.cy-h*.36,w,h,z:z+.1+i*.01,depth:1,group:'head',phys:null,fade:'mouthOpen',mouthAnchor:box,img:mouthArt(expression)}));
  }
  function parts(rig){
    const hair=rig.layers.filter(L=>/^(front hair|back hair|side hair)(_|$)/.test(L.name));
    const crown=Math.min(rig.anchors.face.y0,...hair.map(L=>L.y));
    const rgb=palette(rig,0),ear=art('ear',rgb),tail=art('tail',rgb);
    const companions=[{name:'companion tail',kind:'tail',img:tail,z:-3,group:'body'},
      {name:'companion ear L',kind:'ear',earSide:-1,img:ear,z:-2,group:'head'},
      {name:'companion ear R',kind:'ear',earSide:1,img:ear,z:-1,group:'head'}].map(L=>({...L,
        source:'Optional '+L.name,crown,x:0,y:0,w:L.img.width,h:L.img.height,depth:1,phys:null,fade:null,companion:true}));
    return companions.concat(mouthParts(rig));
  }
  return {idlePose,idleHead,TURN_TILT,blink,breath,mouth,response,nodPose,nodPoint,ROLL,rollPoint,collarWeight,shoulderLift,turnShift,torsoPoint,hatWeight,rowExtents,tailPoint,earPoint,palette,art,parts,mouthWeights,mouthBox};
});
