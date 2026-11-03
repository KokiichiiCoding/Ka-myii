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
  // Ear and tail designs, by the value of the Animal ears / Tail selects (0 is off).
  const EARS=[null,'cat','fox','wolf','bunny','bear'],TAILS=[null,'cat','fox','wolf','devil'];
  // How far each ear leans out from the head (radians): bunny ears stand up, bear ears sit round.
  const EAR_TILT={cat:.14,fox:.16,wolf:.12,bunny:.05,bear:.32};
  // Ears are 128x320 with the root at y 311; tails 128x384 with the root at y 380. ``top`` is
  // where the drawing starts, so the stage only makes room for what is drawn.
  function art(kind,rgb,style){
    const canvas=document.createElement('canvas');canvas.width=128;canvas.height=kind==='tail'?384:320;
    const c=canvas.getContext('2d'),colour=(amount)=>'rgb('+rgb.map(v=>Math.round(clamp(v+amount,0,255))).join(',')+')';
    c.lineJoin='round';c.lineCap='round';
    const ink=colour(-75),path=draw=>{const p=new Path2D();draw(p);p.closePath();return p;};
    const strokes=(colour,width,lines)=>{c.strokeStyle=colour;c.lineWidth=width;for(const l of lines){c.beginPath();c.moveTo(l[0],l[1]);c.quadraticCurveTo(l[2],l[3],l[4],l[5]);c.stroke();}};
    const gradient=(stops,x0,y0,x1,y1)=>{const g=c.createLinearGradient(x0,y0,x1,y1);stops.forEach(([at,amount])=>g.addColorStop(at,colour(amount)));return g;};
    let top=0;
    if(kind==='tail'&&(style==='fox'||style==='wolf')){
      const fox=style==='fox',outline=path(p=>{
        if(fox){p.moveTo(50,382);p.bezierCurveTo(30,330,4,250,8,170);p.bezierCurveTo(10,90,40,30,64,8);p.bezierCurveTo(88,30,118,90,120,170);p.bezierCurveTo(124,250,98,330,78,382);}
        else{p.moveTo(52,382);p.bezierCurveTo(36,320,16,240,18,160);p.bezierCurveTo(20,84,44,26,64,10);p.bezierCurveTo(84,26,108,84,110,160);p.bezierCurveTo(112,240,92,320,76,382);}});
      c.fillStyle=gradient([[0,-38],[.45,10],[1,-26]],0,0,128,0);c.fill(outline);
      c.save();c.clip(outline);
      c.fillStyle=fox?'#f8f1e8':colour(-70);   // a fox's white tip, a wolf's dark one
      c.beginPath();c.moveTo(0,0);c.lineTo(128,0);c.lineTo(128,92);c.bezierCurveTo(104,118,84,82,64,108);c.bezierCurveTo(44,82,24,118,0,92);c.fill();
      strokes(colour(26),3,[[40,350,22,240,34,130],[64,360,58,230,64,120],[88,350,106,240,94,130]]);
      c.restore();c.strokeStyle=ink;c.lineWidth=3;c.stroke(outline);top=6;
    }else if(kind==='tail'&&style==='devil'){
      // A thin dark tail with a spade tip; it keeps its own colours rather than the fur's.
      const shaft=path(p=>{p.moveTo(57,382);p.bezierCurveTo(58,300,59,170,60,92);p.lineTo(68,92);p.bezierCurveTo(69,170,70,300,71,382);});
      const spade=path(p=>{p.moveTo(64,8);p.bezierCurveTo(92,30,118,62,100,92);p.quadraticCurveTo(84,104,64,88);p.quadraticCurveTo(44,104,28,92);p.bezierCurveTo(10,62,36,30,64,8);});
      for(const p of [shaft,spade]){c.fillStyle='#3d2639';c.fill(p);c.strokeStyle='#1b1018';c.lineWidth=3;c.stroke(p);}
      strokes('#7a4166',3,[[50,30,36,58,38,82],[61,370,61,240,62,110]]);top=6;
    }else if(kind==='tail'){
      const g=c.createLinearGradient(32,0,96,0);g.addColorStop(0,colour(-38));g.addColorStop(.45,colour(8));g.addColorStop(1,colour(-20));
      c.beginPath();c.moveTo(38,380);c.bezierCurveTo(32,280,32,100,43,34);c.quadraticCurveTo(64,-13,85,34);c.bezierCurveTo(96,100,96,280,90,380);c.closePath();
      c.fillStyle=g;c.fill();c.strokeStyle=ink;c.lineWidth=3;c.stroke();
      c.save();c.clip();c.fillStyle=colour(30);c.beginPath();c.ellipse(64,38,35,36,0,0,Math.PI*2);c.fill();c.restore();
      c.beginPath();c.moveTo(48,320);c.quadraticCurveTo(41,190,53,85);c.strokeStyle=colour(24);c.lineWidth=3;c.stroke();
    }else if(style==='fox'||style==='wolf'){
      const fox=style==='fox',tip=fox?96:118;
      const outline=path(p=>{
        if(fox){p.moveTo(6,309);p.bezierCurveTo(10,240,34,150,60,104);p.quadraticCurveTo(66,94,72,105);p.bezierCurveTo(98,152,120,240,122,309);}
        else{p.moveTo(14,309);p.bezierCurveTo(16,250,36,172,58,128);p.quadraticCurveTo(64,116,70,128);p.bezierCurveTo(92,172,112,250,114,309);}
        for(const [x,y] of [[108,300],[98,311],[86,301],[74,312],[62,302],[50,312],[38,301],[26,311],[16,300]])p.lineTo(x,y);});   // fur at the base
      c.fillStyle=gradient([[0,fox?6:-30],[.55,8],[1,-32]],0,tip,0,311);c.fill(outline);
      c.save();c.clip(outline);
      if(fox){c.fillStyle='rgb(52,38,42)';c.beginPath();c.moveTo(0,158);c.quadraticCurveTo(64,186,128,158);c.lineTo(128,0);c.lineTo(0,0);c.fill();}
      const inner=path(p=>{p.moveTo(28,292);p.quadraticCurveTo(40,206,64,fox?146:168);p.quadraticCurveTo(88,206,100,292);
        for(const [x,y] of [[90,280],[82,292],[73,279],[64,293],[55,279],[46,292],[38,280]])p.lineTo(x,y);});
      c.fillStyle=fox?'#f5ede3':colour(58);c.fill(inner);c.strokeStyle=fox?'#cdbdaa':colour(-12);c.lineWidth=2;c.stroke(inner);
      strokes(fox?'#fffaf3':colour(84),4,[[40,286,46,250,56,222],[88,286,82,250,72,222],[64,292,63,262,64,236]]);
      c.restore();c.strokeStyle=ink;c.lineWidth=3;c.stroke(outline);top=tip-4;
    }else if(style==='bunny'){
      const outline=path(p=>{p.moveTo(36,311);p.bezierCurveTo(16,220,20,40,64,12);p.bezierCurveTo(108,40,112,220,92,311);p.quadraticCurveTo(64,300,36,311);});
      c.fillStyle=gradient([[0,20],[1,-30]],0,12,0,311);c.fill(outline);c.strokeStyle=ink;c.lineWidth=3;c.stroke(outline);
      const inner=path(p=>{p.moveTo(49,296);p.bezierCurveTo(36,212,40,74,64,42);p.bezierCurveTo(88,74,92,212,79,296);p.quadraticCurveTo(64,288,49,296);});
      c.fillStyle='#eaabb6';c.fill(inner);c.strokeStyle='#c4808f';c.lineWidth=2;c.stroke(inner);
      strokes('#f7cdd4',4,[[56,270,48,170,58,80]]);top=8;
    }else if(style==='bear'){
      const outline=path(p=>{p.moveTo(14,311);p.bezierCurveTo(-4,250,20,188,64,188);p.bezierCurveTo(108,188,132,250,114,311);p.quadraticCurveTo(64,300,14,311);});
      c.fillStyle=gradient([[0,18],[1,-30]],0,188,0,311);c.fill(outline);c.strokeStyle=ink;c.lineWidth=3;c.stroke(outline);
      c.beginPath();c.ellipse(64,266,31,34,0,0,Math.PI*2);c.fillStyle=colour(46);c.fill();c.strokeStyle=colour(-24);c.lineWidth=2;c.stroke();
      top=184;
    }else{
      c.translate(0,160);top=160;   // the original cat ear, drawn in the bottom half
      const g=c.createLinearGradient(20,30,108,155);g.addColorStop(0,colour(22));g.addColorStop(1,colour(-35));
      c.beginPath();c.moveTo(9,151);c.bezierCurveTo(11,109,24,42,52,9);c.quadraticCurveTo(61,-2,69,11);c.bezierCurveTo(100,49,116,106,119,151);c.quadraticCurveTo(101,140,93,151);c.lineTo(83,142);c.lineTo(74,153);c.lineTo(63,145);c.lineTo(51,155);c.lineTo(39,145);c.quadraticCurveTo(21,143,9,151);c.closePath();
      c.fillStyle=g;c.fill();c.strokeStyle=ink;c.lineWidth=3;c.stroke();
      c.beginPath();c.moveTo(30,128);c.quadraticCurveTo(39,62,62,31);c.quadraticCurveTo(85,65,99,128);c.quadraticCurveTo(62,105,30,128);
      c.fillStyle='#ce939e';c.fill();c.strokeStyle='#a66f83';c.lineWidth=2;c.stroke();
      c.beginPath();c.moveTo(40,118);c.quadraticCurveTo(48,77,61,51);c.strokeStyle='#e8b3bd';c.lineWidth=4;c.stroke();
      c.beginPath();c.moveTo(22,131);c.lineTo(37,119);c.moveTo(101,131);c.lineTo(89,118);c.strokeStyle=colour(28);c.lineWidth=4;c.stroke();
    }
    const image=c.getImageData(0,0,canvas.width,canvas.height);image.top=top;return image;
  }
  // Face marks drawn in the browser, so every character has them without painting (painted
  // toggles match the art style better). app.js markBox() places them on the face anchors.
  const MARKS={blush:[128,72],gloom:[256,192],anger:[96,96],sweat:[64,96]};
  function markArt(mark){
    const [w,h]=MARKS[mark],canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;
    const c=canvas.getContext('2d');c.lineCap='round';c.lineJoin='round';
    if(mark==='blush'){
      c.save();c.scale(1,h/w);
      const g=c.createRadialGradient(w/2,w/2,0,w/2,w/2,w/2);g.addColorStop(0,'rgba(255,96,124,.6)');g.addColorStop(.55,'rgba(255,112,140,.34)');g.addColorStop(1,'rgba(255,130,155,0)');
      c.fillStyle=g;c.fillRect(0,0,w,w);c.restore();
      c.strokeStyle='rgba(214,58,92,.75)';c.lineWidth=3;
      for(let i=0;i<4;i++){c.beginPath();c.moveTo(w*(.27+i*.13),h*.64);c.lineTo(w*(.35+i*.13),h*.36);c.stroke();}
    }else if(mark==='gloom'){
      const g=c.createLinearGradient(0,0,0,h);g.addColorStop(0,'rgba(34,28,92,0)');g.addColorStop(.2,'rgba(34,28,92,.66)');g.addColorStop(.62,'rgba(48,42,110,.4)');g.addColorStop(1,'rgba(60,60,124,0)');
      c.fillStyle=g;c.fillRect(0,0,w,h);
      c.globalCompositeOperation='destination-in';   // soft sides: a shadow on the face, not a box
      const side=c.createLinearGradient(0,0,w,0);side.addColorStop(0,'rgba(0,0,0,0)');side.addColorStop(.2,'#000');side.addColorStop(.8,'#000');side.addColorStop(1,'rgba(0,0,0,0)');
      c.fillStyle=side;c.fillRect(0,0,w,h);c.globalCompositeOperation='source-over';
      c.lineWidth=4;
      for(const x of [.24,.33,.42,.51,.6,.69,.78]){   // lines fading in from the top
        const y0=h*(.08+.05*Math.abs(Math.cos(x*23))),y1=h*(.4+.14*Math.abs(Math.sin(x*17))),line=c.createLinearGradient(0,y0,0,y1);
        line.addColorStop(0,'rgba(28,22,78,0)');line.addColorStop(.3,'rgba(28,22,78,.7)');line.addColorStop(1,'rgba(28,22,78,.45)');
        c.strokeStyle=line;c.beginPath();c.moveTo(x*w,y0);c.lineTo(x*w,y1);c.stroke();
      }
    }else if(mark==='anger'){
      // Four arcs around a gap, bulging to the middle, like the drawn Anger vein toggle.
      const d=w*.2,r=w*.19;
      for(const [colour,width] of [['#5f1423',w*.14],['#e43048',w*.085]]){c.strokeStyle=colour;c.lineWidth=width;
        for(const [sx,sy,start] of [[-1,-1,0],[1,-1,.5],[1,1,1],[-1,1,1.5]]){c.beginPath();c.arc(w/2+sx*d,h/2+sy*d,r,(start+.08)*Math.PI,(start+.42)*Math.PI);c.stroke();}}
    }else{
      const r=w*.32,cx=w/2,cy=h-r-6,tip=6,b=Math.acos(Math.min(1,r/(cy-tip)));
      c.beginPath();c.moveTo(cx,tip);c.arc(cx,cy,r,b-Math.PI/2,1.5*Math.PI-b);c.closePath();
      c.fillStyle='#c3e4ff';c.fill();c.strokeStyle='#3c69aa';c.lineWidth=3;c.stroke();
      c.beginPath();c.ellipse(cx-r*.38,cy-r*.2,r*.17,r*.28,0,0,Math.PI*2);c.fillStyle='#fff';c.fill();
    }
    return c.getImageData(0,0,w,h);
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
    const rgb=palette(rig,0),ear=art('ear',rgb,'cat'),tail=art('tail',rgb,'cat');
    const companions=[{name:'companion tail',kind:'tail',img:tail,z:-3,group:'body'},
      {name:'companion ear L',kind:'ear',earSide:-1,img:ear,z:-2,group:'head'},
      {name:'companion ear R',kind:'ear',earSide:1,img:ear,z:-1,group:'head'}].map(L=>({...L,
        source:'Optional '+L.name,crown,top:L.img.top,x:0,y:0,w:L.img.width,h:L.img.height,depth:1,phys:null,fade:null,companion:true}));
    // Blush sits on the cheeks under bangs, nose and mouth; the rest float over the hair.
    const face=rig.layers.find(L=>L.name==='face'),high=Math.max(0,...rig.layers.filter(L=>L.group==='head').map(L=>L.z));
    const marks=[['blush',-1,'L',(face?face.z:high)+.05],['blush',1,'R',(face?face.z:high)+.06],['gloom',0,'',high+.2],
      ['anger',0,'',high+.3],['sweat',0,'',high+.31]].map(([mark,side,suffix,z])=>{const img=markArt(mark);
      return {name:('mark '+mark+' '+suffix).trim(),source:'Instant '+mark,kind:'mark',mark,markSide:side,param:'mark'+mark[0].toUpperCase()+mark.slice(1),
        img,z,x:0,y:0,w:img.width,h:img.height,depth:1,group:'head',phys:null,fade:null,companion:true};});
    return companions.concat(marks,mouthParts(rig));
  }
  // One-shot gestures added to the pose targets (angles in rig units, -1..1). They start and
  // end at zero, so they layer over idle, pointer or camera motion without a snap.
  const GESTURES={
    nod:{duration:1.2,pose:(t,w)=>({y:-.55*w*(1-Math.cos(Math.min(t,1.1)*Math.PI*2/.55))/2,body:0})},
    shake:{duration:1.4,pose:(t,w)=>({x:.6*w*Math.sin(t*Math.PI*2*2.1)*(1-t/1.8),y:-.08*w})},
    tilt:{duration:1.8,pose:(t,w)=>({z:.6*w,x:.12*w,brow:.7*w,eyeY:.2*w})},
    yay:{duration:1.4,pose:(t,w)=>({y:.35*w*Math.abs(Math.sin(t*Math.PI*2.6)),mouthForm:1.2*w,brow:.4*w,body:.12*w*Math.sin(t*Math.PI*2.6)})},
    shy:{duration:2.2,pose:(t,w)=>({x:-.55*w,y:-.3*w,z:-.2*w,eyeX:-.5*w,eyeY:-.45*w,mouthForm:.4*w})},
  };
  function gesture(name,seconds){
    const g=GESTURES[name];
    if(!g||!(seconds>=0)||seconds>=g.duration)return null;
    const w=smooth(seconds/.18)*(1-smooth((seconds-(g.duration-.35))/.35));
    return Object.assign({x:0,y:0,z:0,body:0,brow:0,mouthForm:0,eyeX:0,eyeY:0},g.pose(seconds,w));
  }
  // Keep the deformation attached to the original artwork, not to coordinates already
  // displaced by head/shoulder motion. Cache the returned weights until the anchor moves.
  function chestWeights(base, chest, heightOffset){
    const result=new Float32Array(base.length/2),cy=chest.cy+heightOffset;
    for(let i=0;i<result.length;i++){
      const x=(base[i*2]-chest.cx)/chest.rx,y=(base[i*2+1]-cy)/chest.ry;
      result[i]=Math.exp(-x*x-y*y);
    }
    return result;
  }
  function chestOffset(offset,strength,faceHeight){
    if(!Number.isFinite(offset)||!Number.isFinite(strength)||strength<=0)return 0;
    const limit=Math.max(1,faceHeight*.12);
    return limit*Math.tanh(offset*strength/limit);
  }
  return {gesture,GESTURES:Object.keys(GESTURES),idlePose,idleHead,TURN_TILT,blink,breath,mouth,response,nodPose,nodPoint,ROLL,rollPoint,collarWeight,shoulderLift,turnShift,torsoPoint,hatWeight,rowExtents,tailPoint,earPoint,palette,art,markArt,EARS,TAILS,EAR_TILT,parts,mouthWeights,mouthBox,chestWeights,chestOffset};
});
