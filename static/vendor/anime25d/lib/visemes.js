/* Ka-myii: Japanese vowels (A I U E O) from microphone audio, for painted viseme mouths. MIT.
   Linear prediction (LPC) finds the first two formants of a ~40 ms frame; the vowel is the
   nearest of five reference formant pairs (log-frequency distance), softened into weights so
   mouths crossfade. The references are typical adult male Japanese vowels, scaled up with the
   frame's pitch (a higher voice has higher formants: a man's I overlaps a woman's E otherwise);
   no per-speaker training. */
(function(root,factory){
  if(typeof module==='object'&&module.exports)module.exports=factory();
  else root.RigVisemes=factory();
})(typeof self!=='undefined'?self:this,function(){
  'use strict';
  const VOWELS=['a','i','u','e','o'];
  const FORMANTS={a:[750,1180],i:[300,2200],u:[340,1300],e:[470,1900],o:[480,850]};
  const RATE=12000,ORDER=12,BASE_PITCH=120;
  // Decimate to ~12 kHz (box-filtered): formants lie below 4 kHz.
  function decimate(samples,sampleRate){
    const step=Math.max(1,Math.round(sampleRate/RATE)),n=Math.floor(samples.length/step),x=new Float64Array(n);
    for(let i=0;i<n;i++){let s=0;for(let j=0;j<step;j++)s+=samples[i*step+j];x[i]=s/step;}
    return {x,rate:sampleRate/step};
  }
  // Pre-emphasised, Hamming-windowed copy for LPC.
  function emphasise(x){
    const y=Float64Array.from(x),n=y.length;
    for(let i=n-1;i>0;i--)y[i]-=.97*y[i-1];
    for(let i=0;i<n;i++)y[i]*=.54-.46*Math.cos(2*Math.PI*i/(n-1));
    return y;
  }
  // Voice pitch (70–400 Hz) from the normalised autocorrelation, or null when unvoiced. The
  // shortest lag near the best peak wins: whole multiples of the period correlate as well.
  function pitch(x,rate){
    const lo=Math.floor(rate/400),hi=Math.min(Math.floor(x.length/2),Math.ceil(rate/70)),r=[];
    for(let k=lo;k<=hi;k++){
      let s=0,e0=0,e1=0;
      for(let i=0;i+k<x.length;i++){s+=x[i]*x[i+k];e0+=x[i]*x[i];e1+=x[i+k]*x[i+k];}
      r.push(s/Math.sqrt(e0*e1||1));
    }
    const best=Math.max(...r);
    if(!(best>.3))return null;
    for(let i=1;i<r.length-1;i++)if(r[i]>=.85*best&&r[i]>=r[i-1]&&r[i]>=r[i+1])return rate/(lo+i);
    return null;
  }
  // Levinson-Durbin on the autocorrelation: A(z) = 1 + a1 z^-1 + ... + ap z^-p.
  function lpc(x,p){
    const r=new Float64Array(p+1);
    for(let k=0;k<=p;k++){let s=0;for(let i=k;i<x.length;i++)s+=x[i]*x[i-k];r[k]=s;}
    if(r[0]<=0)return null;
    r[0]*=1.0001;   // a little white noise keeps the recursion stable
    let a=new Float64Array(p+1),e=r[0];a[0]=1;
    for(let i=1;i<=p;i++){
      let acc=r[i];for(let j=1;j<i;j++)acc+=a[j]*r[i-j];
      const k=-acc/e,next=a.slice();
      for(let j=1;j<i;j++)next[j]=a[j]+k*a[i-j];
      next[i]=k;a=next;e*=1-k*k;
      if(e<=0)return null;
    }
    return a;
  }
  // Roots of A(z) (Durand-Kerner): each resonance is a root pair at angle 2πF/rate, with a
  // bandwidth from its radius. Envelope peak-picking merged close formants; roots do not.
  function roots(a){
    const p=a.length-1,re=new Float64Array(p),im=new Float64Array(p);
    for(let i=0;i<p;i++){const t=2*Math.PI*i/p+.4;re[i]=.9*Math.cos(t);im[i]=.9*Math.sin(t);}
    for(let it=0;it<200;it++){
      let moved=0;
      for(let i=0;i<p;i++){
        let nr=1,ni=0;   // A(z)·z^p = z^p + a1 z^(p-1) + … + ap, by Horner
        for(let k=1;k<=p;k++){const r=nr*re[i]-ni*im[i]+a[k],m=nr*im[i]+ni*re[i];nr=r;ni=m;}
        let dr=1,di=0;
        for(let j=0;j<p;j++)if(j!==i){const xr=re[i]-re[j],xi=im[i]-im[j],r=dr*xr-di*xi,m=dr*xi+di*xr;dr=r;di=m;}
        const den=dr*dr+di*di||1e-30,qr=(nr*dr+ni*di)/den,qi=(ni*dr-nr*di)/den;
        re[i]-=qr;im[i]-=qi;moved=Math.max(moved,Math.abs(qr)+Math.abs(qi));
      }
      if(moved<1e-10)break;
    }
    return Array.from(re,(r,i)=>[r,im[i]]);
  }
  // First two formants [F1, F2] in Hz (resonances 200 Hz–3.5 kHz narrower than 500 Hz), or null.
  function formants(samples,sampleRate){
    const {x,rate}=decimate(samples,sampleRate),a=lpc(emphasise(x),ORDER);
    if(!a)return null;
    const found=[];
    for(const [r,i] of roots(a)){
      if(i<=0)continue;
      const f=Math.atan2(i,r)*rate/(2*Math.PI),bw=-Math.log(Math.hypot(r,i))*rate/Math.PI;
      if(f>=200&&f<=3500&&bw>0&&bw<500)found.push(f);
    }
    found.sort((p,q)=>p-q);
    const f1=found.find(f=>f>=240&&f<=1050);
    if(f1==null)return null;
    const f2=found.find(f=>f>=Math.max(700,f1+180));
    return f2==null?null:[f1,f2];
  }
  // Soft vowel weights from [F1, F2] (sum to 1) at voice pitch `f0` (Hz, or null: in between).
  function weights(f,f0,sharp){
    const out={},k=sharp||18,s=Math.min(1.35,Math.max(.95,Math.pow((f0||170)/BASE_PITCH,.25)));
    let total=0;
    for(const v of VOWELS){
      // F1 counts half: at high pitch LPC pulls it onto a harmonic; F2 separates the vowels.
      const d=Math.hypot(.5*Math.log(f[0]/(FORMANTS[v][0]*s)),Math.log(f[1]/(FORMANTS[v][1]*s)));
      total+=out[v]=Math.exp(-k*d*d);
    }
    for(const v of VOWELS)out[v]/=total||1;
    return out;
  }
  function classify(samples,sampleRate){
    const f=formants(samples,sampleRate);
    if(!f)return null;
    const {x,rate}=decimate(samples,sampleRate);
    return weights(f,pitch(x,rate));
  }
  // From the face tracker's mouth shape when there is no audio: wide → I/E, round → U/O.
  function fromShape(open,form,round){
    const w={a:0,i:0,u:0,e:0,o:0},small=open<.45;
    if(round>.4||form<-.3)w[small?'u':'o']=1;else if(form>.3)w[small?'i':'e']=1;else w.a=1;
    return w;
  }
  function blend(cur,target,dt,rate){
    const k=1-Math.exp(-Math.max(0,dt)*(rate||18)),out={};
    for(const v of VOWELS)out[v]=(cur[v]||0)+((target[v]||0)-(cur[v]||0))*k;
    return out;
  }
  // Layer alphas for weights drawn bottom to top in `order`: each weight over the running total,
  // so "over" compositing gives exactly the weighted mix at full coverage (two half-weights
  // drawn at 50% each looked pale mid-crossfade).
  function stackAlpha(w,order){
    const out={};let total=0;
    for(const v of order){const x=w[v]||0;total+=x;out[v]=total>1e-6?x/total:0;}
    return out;
  }
  return {VOWELS,FORMANTS,formants,pitch,weights,classify,fromShape,blend,stackAlpha};
});
