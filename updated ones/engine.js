const SG=(()=>{
const FIELDS=[['timestamp','time date datetime updated synced when','Timestamp',1],['sku','sku product item id code article','Product',1],['channel','channel marketplace platform store source','Channel',1],
['master_price','master_price master price erp base list','Master price',1],['channel_price','channel_price channel price listed shown selling','Channel price',1],
['master_stock','master_stock master stock inventory erp qty quantity','Master stock',0],['channel_stock','channel_stock channel stock available listed','Channel stock',0],
['units_sold','units_sold sold sales orders demand','Units sold',0],['sync_delay_min','sync_delay delay minutes lag latency since','Minutes since sync',0],['sync_attempts','sync_attempts attempts retries tries','Sync attempts',0],['sync_status','sync_status status result outcome state','Sync status',0],['is_anomaly','is_anomaly anomaly label error flag truth','Real-error label (optional)',0]];
const FEAT=['price_diff','stock_diff','sync_delay','stock_cover','price_change','stock_change','sync_attempts','update_gap','sync_status'];
const FL={price_diff:'Price gap',stock_diff:'Stock gap',sync_delay:'Sync delay',stock_cover:'Stock cover',price_change:'Price jump',stock_change:'Stock jump',sync_attempts:'Sync retries',update_gap:'Update gap',sync_status:'Sync status'};
function parseCSV(t){t=t.replace(/^\uFEFF/,'');const first=t.slice(0,t.indexOf('\n')>0?t.indexOf('\n'):t.length);const d=[',',';','\t','|'].map(c=>[c,first.split(c).length]).sort((a,b)=>b[1]-a[1])[0][0];
 const rows=[];let row=[],f='',q=false;for(let i=0;i<t.length;i++){const c=t[i];
  if(q){if(c==='"'){if(t[i+1]==='"'){f+='"';i++;}else q=false;}else f+=c;}
  else if(c==='"')q=true;else if(c===d){row.push(f);f='';}else if(c==='\n'||c==='\r'){if(c==='\r'&&t[i+1]==='\n')i++;row.push(f);f='';if(row.length>1||row[0]!=='')rows.push(row);row=[];}else f+=c;}
 if(f!==''||row.length){row.push(f);rows.push(row);}return{head:rows[0].map(h=>h.trim()),rows:rows.slice(1)};}
function guess(head){const norm=h=>h.toLowerCase().replace(/[^a-z0-9]+/g,' ').trim(),H=head.map(norm),used=new Set(),m={};
 FIELDS.forEach(([k,syn])=>{let best=-1,bs=0;H.forEach((h,i)=>{if(used.has(i))return;let s=0;if(h===k.replace(/_/g,' '))s=10;syn.split(' ').forEach(w=>{if(h.includes(w))s+=1;});if(k.startsWith('master')&&!h.includes('master'))s-=.5;if(k.startsWith('channel_')&&h.includes('master'))s-=2;if(s>bs){bs=s;best=i;}});
  if(best>=0&&bs>=1){m[k]=best;used.add(best);}});return m;}
const num=v=>{if(v==null)return NaN;if(typeof v==='number')return v;let s=String(v).trim().replace(/[^0-9eE.,\-]/g,'');if(s==='')return NaN;if(/,\d{1,2}$/.test(s)&&!/\./.test(s))s=s.replace(',','.');else s=s.replace(/,/g,'');return parseFloat(s);};
const tim=v=>{if(v==null||v==='')return NaN;const n=Number(v);if(!isNaN(n)&&String(v).trim()!==''){return n>1e11?n:n>1e8?n*1000:NaN;}const t=Date.parse(String(v).replace(' ','T'));return t;};
const med=a=>{const s=[...a].sort((x,y)=>x-y);return s.length?s[s.length>>1]:NaN;};
const pctl=(a,p)=>{const s=Float64Array.from(a).sort();return s[Math.min(s.length-1,Math.floor(p/100*s.length))];};
function clean(head,rows,map,opt={}){const rep={rows_in:rows.length,unusable:0,duplicates:0,invalid:0,filled:{}};const sk=new Map(),ch=new Map(),R=[];
 const g=(r,k)=>map[k]!=null?r[map[k]]:undefined,seen=new Set();
 for(const r of rows){const ts=tim(g(r,'timestamp')),s=String(g(r,'sku')??'').trim(),c=String(g(r,'channel')??'').trim(),mp=num(g(r,'master_price'));
  if(isNaN(ts)||!s||!c||!(mp>0)){rep.unusable++;continue;}const key=s+'|'+c+'|'+ts;if(seen.has(key)){rep.duplicates++;continue;}seen.add(key);
  if(!sk.has(s))sk.set(s,sk.size);if(!ch.has(c))ch.set(c,ch.size);
  const at=map.sync_attempts!=null?Math.max(1,num(g(r,'sync_attempts'))||1):1,sraw=map.sync_status!=null?String(g(r,'sync_status')??'').trim().toLowerCase():'',ss=/^(retry|retrying|warn|warning)$/.test(sraw)?1:/^(fail|failed|error|failure|timeout)$/.test(sraw)?2:0;
  let cp=num(g(r,'channel_price')),ms=num(g(r,'master_stock')),cs=num(g(r,'channel_stock')),us=num(g(r,'units_sold')),sd=num(g(r,'sync_delay_min'));
  if(cp<=0||cs<0||sd<0||us<0){rep.invalid++;if(cp<=0)cp=NaN;if(cs<0)cs=NaN;if(sd<0)sd=NaN;if(us<0)us=NaN;}
  const y=map.is_anomaly!=null?(/^(1|true|yes|y|anomaly)$/i.test(String(g(r,'is_anomaly')).trim())?1:0):-1;
  R.push({t:ts,s:sk.get(s),c:ch.get(c),mp,cp,ms,cs,us,sd,y,at,ss});}
 const has={stock:map.channel_stock!=null&&map.master_stock!=null,units:map.units_sold!=null,delay:map.sync_delay_min!=null,label:map.is_anomaly!=null,attempts:map.sync_attempts!=null,status:map.sync_status!=null},nc=Math.max(ch.size,1);
 const ratio=[...ch.keys()].map((_,i)=>med(R.filter(r=>r.c===i&&r.cp>0).map(r=>r.cp/r.mp)));const dmed=[...ch.keys()].map((_,i)=>med(R.filter(r=>r.c===i&&r.sd>=0).map(r=>r.sd)));
 for(const r of R){if(isNaN(r.cp)){r.cp=+(r.mp*(ratio[r.c]||1)).toFixed(2);rep.filled.channel_price=(rep.filled.channel_price||0)+1;}
  if(has.stock&&isNaN(r.cs)){r.cs=Math.round(r.ms/nc);rep.filled.channel_stock=(rep.filled.channel_stock||0)+1;}if(has.stock&&isNaN(r.ms)){r.ms=r.cs*nc;}
  if(has.delay&&isNaN(r.sd)){r.sd=dmed[r.c]||0;rep.filled.sync_delay_min=(rep.filled.sync_delay_min||0)+1;}if(has.units&&isNaN(r.us))r.us=0;}
 R.sort((a,b)=>a.s-b.s||a.c-b.c||a.t-b.t);rep.rows_out=R.length;
 return{R,sku:[...sk.keys()],ch:[...ch.keys()],has,rep,ratio};}
function fv(C,names,r,pv){const same=pv&&pv.s===r.s&&pv.c===r.c,nc=C.ch.length,h=C.has,f={},cl=x=>Math.min(3,Math.max(-1,isFinite(x)?x:0));f.price_diff=(r.cp/r.mp)/(C.ratio[r.c]||1)-1;
 f.price_change=same?cl((r.cp/pv.cp-1)-(r.mp/pv.mp-1)):0;
 if(h.stock){const fair=r.ms/nc;f.stock_diff=(r.cs-fair)/(fair+1);f.stock_change=same&&pv.cs>0&&pv.ms>0?cl((r.cs/pv.cs-1)-(r.ms/pv.ms-1)):0;if(h.units)f.stock_cover=Math.log1p(Math.max(r.cs,0)/Math.max(r.us,1));}
 if(h.delay)f.sync_delay=Math.log1p(r.sd);if(h.attempts)f.sync_attempts=Math.log1p(r.at);if(h.status)f.sync_status=r.ss;
 f.update_gap=Math.log1p(same?Math.max(0,(r.t-pv.t)/6e4):60);return names.map(n=>f[n]);}
function features(C){const{R,has}=C,names=['price_diff','price_change'];if(has.stock){names.push('stock_diff','stock_change');if(has.units)names.push('stock_cover');}if(has.delay)names.push('sync_delay');if(has.attempts)names.push('sync_attempts');if(has.status)names.push('sync_status');names.push('update_gap');
 const m=names.length,X=new Float64Array(R.length*m);R.forEach((r,i)=>{fv(C,names,r,i?R[i-1]:null).forEach((x,j)=>X[i*m+j]=x);});return{X,names,m};}
function* ae(Z,n,m,h,mask,ep,seed,sub,p0,p1){let st=seed;const rnd=()=>(st=(st*16807)%2147483647)/2147483647,idx=[];for(let i=0;i<n;i++)if(!mask||mask[i])idx.push(i);
 for(let i=idx.length-1;i>0;i--){const k=Math.floor(rnd()*(i+1));[idx[i],idx[k]]=[idx[k],idx[i]];}const tr=idx.slice(0,sub);
 const W1=Float64Array.from({length:m*h},()=>(rnd()-.5)*2*Math.sqrt(6/(m+h))),b1=new Float64Array(h),W2=Float64Array.from({length:h*m},()=>(rnd()-.5)*2*Math.sqrt(6/(m+h))),b2=new Float64Array(m);
 const P=[W1,b1,W2,b2],G=P.map(p=>new Float64Array(p.length)),M1=P.map(p=>new Float64Array(p.length)),V1=P.map(p=>new Float64Array(p.length)),a=new Float64Array(h),o=new Float64Array(m),da=new Float64Array(h);let t=0;
 for(let e=0;e<ep;e++){for(let i=tr.length-1;i>0;i--){const k=Math.floor(rnd()*(i+1));[tr[i],tr[k]]=[tr[k],tr[i]];}
  for(let b=0;b<tr.length;b+=64){G.forEach(g=>g.fill(0));const B=Math.min(64,tr.length-b);
   for(let q=0;q<B;q++){const r=tr[b+q]*m;for(let k=0;k<h;k++){let v=b1[k];for(let j=0;j<m;j++)v+=Z[r+j]*W1[j*h+k];a[k]=Math.tanh(v);}
    for(let j=0;j<m;j++){let v=b2[j];for(let k=0;k<h;k++)v+=a[k]*W2[k*m+j];o[j]=2*(v-Z[r+j])/m;G[3][j]+=o[j];}
    for(let k=0;k<h;k++){let d=0;for(let j=0;j<m;j++){G[2][k*m+j]+=a[k]*o[j];d+=W2[k*m+j]*o[j];}da[k]=d*(1-a[k]*a[k]);G[1][k]+=da[k];for(let j=0;j<m;j++)G[0][j*h+k]+=Z[r+j]*da[k];}}
   t++;const c1=1-Math.pow(.9,t),c2=1-Math.pow(.999,t);P.forEach((p,i)=>{for(let k=0;k<p.length;k++){const g=G[i][k]/B;M1[i][k]=.9*M1[i][k]+.1*g;V1[i][k]=.999*V1[i][k]+.001*g*g;p[k]-=.01*(M1[i][k]/c1)/(Math.sqrt(V1[i][k]/c2)+1e-8);}});}yield p0+(p1-p0)*(e+1)/ep;}
 return{W1,b1,W2,b2,m,h};}
function rec(N,z,out){const{W1,b1,W2,b2,m,h}=N;const a=new Float64Array(h);for(let k=0;k<h;k++){let v=b1[k];for(let j=0;j<m;j++)v+=z[j]*W1[j*h+k];a[k]=Math.tanh(v);}for(let j=0;j<m;j++){let v=b2[j];for(let k=0;k<h;k++)v+=a[k]*W2[k*m+j];out[j]=v;}}
function* fitG(F,opt={}){const{X,m}=F,n=X.length/m,trim=opt.trim??3,h=Math.max(2,Math.round(m*.45)),ep=opt.ep??25,
 mu=new Float64Array(m),sd=new Float64Array(m);for(let i=0;i<n;i++)for(let j=0;j<m;j++)mu[j]+=X[i*m+j]/n;
 for(let i=0;i<n;i++)for(let j=0;j<m;j++)sd[j]+=(X[i*m+j]-mu[j])**2/n;for(let j=0;j<m;j++)sd[j]=Math.sqrt(sd[j])||1;
 const Z=new Float64Array(n*m);for(let i=0;i<n;i++)for(let j=0;j<m;j++)Z[i*m+j]=Math.max(-12,Math.min(12,(X[i*m+j]-mu[j])/sd[j]));
 const errs=N=>{const E=new Float32Array(n*m),o=new Float64Array(m);for(let i=0;i<n;i++){rec(N,Z.subarray(i*m,i*m+m),o);for(let j=0;j<m;j++)E[i*m+j]=(Z[i*m+j]-o[j])**2;}return E;};
 const sc=(E,S)=>{const s=new Float32Array(n);for(let i=0;i<n;i++){let b=0;for(let j=0;j<m;j++)b=Math.max(b,E[i*m+j]/S[j]);s[i]=b;}return s;};
 const scale=E=>Array.from({length:m},(_,j)=>pctl(Array.from({length:n},(_,i)=>E[i*m+j]),99)+1e-9);
 let N=yield* ae(Z,n,m,h,null,opt.ep1??ep,7,opt.sub??20000,0,.35),E=errs(N),s0=sc(E,scale(E)),cut=pctl(s0,100-trim),keep=new Uint8Array(n);for(let i=0;i<n;i++)keep[i]=s0[i]<=cut?1:0;
 N=yield* ae(Z,n,m,h,keep,ep,11,opt.sub??20000,.35,1);E=errs(N);const S=scale(E),s=sc(E,S),cl=[];for(let i=0;i<n;i++)if(keep[i])cl.push(s[i]);const thr=pctl(cl,opt.pct??99.6)||1e-9;
 const ratio=new Float32Array(n);for(let i=0;i<n;i++)ratio[i]=s[i]/thr;const En=new Float32Array(n*m);for(let i=0;i<n;i++)for(let j=0;j<m;j++)En[i*m+j]=E[i*m+j]/S[j];
 return{ratio,En,thr,names:F.names,model:{mu,sd,N,S,m}};}
function fit(F,opt){const g=fitG(F,opt);let r;while(!(r=g.next()).done);return r.value;}
function fitAsync(F,opt,cb){return new Promise(res=>{const g=fitG(F,opt),step=()=>{const r=g.next();if(r.done)res(r.value);else{cb&&cb(r.value);setTimeout(step,0);}};setTimeout(step,20);});}
function apply(F,model,thr){const{X,m}=F,n=X.length/m,{mu,sd,N,S}=model,ratio=new Float32Array(n),En=new Float32Array(n*m),z=new Float64Array(m),o=new Float64Array(m);
 for(let i=0;i<n;i++){for(let j=0;j<m;j++)z[j]=Math.max(-12,Math.min(12,(X[i*m+j]-mu[j])/sd[j]));rec(N,z,o);let b=0;for(let j=0;j<m;j++){const e=(z[j]-o[j])**2/S[j];En[i*m+j]=e;if(e>b)b=e;}ratio[i]=b/thr;}
 return{ratio,En,thr,names:F.names,model};}
function pack(f){const mm=f.model,N=mm.N;return{thr:f.thr,mu:Array.from(mm.mu),sd:Array.from(mm.sd),S:Array.from(mm.S),m:mm.m,N:{W1:Array.from(N.W1),b1:Array.from(N.b1),W2:Array.from(N.W2),b2:Array.from(N.b2),m:N.m,h:N.h}};}
function scoreOne(model,x){const{mu,sd,N,S,m}=model,z=x.map((v,j)=>Math.max(-12,Math.min(12,(v-mu[j])/sd[j]))),o=new Float64Array(m);rec(N,z,o);const e=z.map((v,j)=>((v-o[j])**2)/S[j]);return{e,r:Math.max(...e)};}
return{FIELDS,FEAT,FL,parseCSV,guess,clean,features,fv,fit,fitAsync,apply,pack,scoreOne,pctl,med};})();
if(typeof module!=='undefined')module.exports=SG;
