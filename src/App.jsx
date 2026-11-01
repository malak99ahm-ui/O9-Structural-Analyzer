import React, { useMemo, useState } from 'react';
import { Plus, Trash2, Printer, RotateCcw, BarChart3, Layers, Activity, AlertTriangle } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import './style.css';

const num = (v) => Number.isFinite(Number(v)) ? Number(v) : 0;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, num(v)));
const fmt = (v) => Number.isFinite(v) ? v.toFixed(2) : '—';
const uid = () => Date.now() + Math.floor(Math.random() * 100000);

function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    if (Math.abs(M[p][k]) < 1e-10) throw new Error('النظام غير مستقر: تحقق من المساند أو القيود.');
    [M[k], M[p]] = [M[p], M[k]];
    const d = M[k][k];
    for (let j = k; j <= n; j++) M[k][j] /= d;
    for (let i = 0; i < n; i++) if (i !== k) {
      const f = M[i][k];
      for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
    }
  }
  return M.map(r => r[n]);
}

// Euler-Bernoulli beam finite-element model; constant EI, vertical DOF + rotation at each node.
function analyze({ L, supports, pointLoads, udls, triangles }) {
  if (!(L > 0)) throw new Error('طول العتب يجب أن يكون أكبر من صفر.');
  const validSupports = supports.map(s => ({...s, x: clamp(s.x, 0, L)}));
  if (!validSupports.length) throw new Error('أضف مسنداً واحداً على الأقل.');
  const breaks = [0, L, ...validSupports.map(s => s.x),
    ...pointLoads.map(p => clamp(p.x, 0, L)),
    ...udls.flatMap(q => [clamp(q.start,0,L), clamp(q.end,0,L)]),
    ...triangles.flatMap(q => [clamp(q.start,0,L), clamp(q.end,0,L)])];
  const baseN = 100;
  for (let i=0;i<=baseN;i++) breaks.push(L*i/baseN);
  const xs = [...new Set(breaks.map(x => Number(x.toFixed(7))))].sort((a,b)=>a-b);
  const n = xs.length, nd = 2*n;
  const K = Array.from({length:nd},()=>Array(nd).fill(0));
  const F = Array(nd).fill(0);
  const EI = 1;
  for (let e=0;e<n-1;e++) {
    const x1=xs[e], x2=xs[e+1], le=x2-x1;
    const k = [
      [12,6*le,-12,6*le],
      [6*le,4*le*le,-6*le,2*le*le],
      [-12,-6*le,12,-6*le],
      [6*le,2*le*le,-6*le,4*le*le]
    ].map(row=>row.map(v=>v*EI/(le**3)));
    const ids=[2*e,2*e+1,2*e+2,2*e+3];
    for(let a=0;a<4;a++) for(let b=0;b<4;b++) K[ids[a]][ids[b]]+=k[a][b];
    const mid=(x1+x2)/2;
    let q1=0,q2=0;
    for(const q of udls) if(mid>=Math.min(q.start,q.end) && mid<=Math.max(q.start,q.end)) { q1+=num(q.w); q2+=num(q.w); }
    for(const q of triangles) {
      const a=Math.min(q.start,q.end), b=Math.max(q.start,q.end);
      if(mid>=a && mid<=b && b>a) {
        const leftIsStart = q.start <= q.end;
        const wa=leftIsStart?num(q.w1):num(q.w2), wb=leftIsStart?num(q.w2):num(q.w1);
        const at=(x)=>wa+(wb-wa)*(x-a)/(b-a);
        q1+=at(x1); q2+=at(x2);
      }
    }
    // Downward distributed loads; equivalent nodal actions.
    const fe=[-le*(7*q1+3*q2)/20, -le*le*(3*q1+2*q2)/60,
              -le*(3*q2+7*q1)/20, le*le*(2*q2+3*q1)/60];
    for(let a=0;a<4;a++) F[ids[a]]+=fe[a];
  }
  for(const p of pointLoads) {
    const x=clamp(p.x,0,L), idx=xs.findIndex(v=>Math.abs(v-x)<1e-6);
    if(idx>=0) F[2*idx]-=num(p.p);
  }
  const constrained = new Set();
  for(const s of validSupports) {
    const idx=xs.findIndex(v=>Math.abs(v-s.x)<1e-6);
    if(idx<0) continue;
    constrained.add(2*idx); // pin and roller restrain vertical displacement
    if(s.type==='fixed') constrained.add(2*idx+1);
  }
  const free=Array.from({length:nd},(_,i)=>i).filter(i=>!constrained.has(i));
  if(!free.length) throw new Error('القيود غير صالحة للتحليل.');
  const Kr=free.map(i=>free.map(j=>K[i][j]));
  const Fr=free.map(i=>F[i]);
  const ur=solveLinear(Kr,Fr);
  const u=Array(nd).fill(0); free.forEach((d,i)=>u[d]=ur[i]);
  const residual=K.map((row,i)=>row.reduce((sum,v,j)=>sum+v*u[j],0)-F[i]);
  const reactions=validSupports.map(s=>{
    const idx=xs.findIndex(v=>Math.abs(v-s.x)<1e-6);
    return {id:s.id,x:s.x,vertical:residual[2*idx],moment:s.type==='fixed'?residual[2*idx+1]:0};
  });
  const data=xs.map(x=>{
    let shear=0, moment=0;
    for(const r of reactions) if(x>=r.x-1e-8) { shear+=r.vertical; moment+=r.vertical*(x-r.x)+r.moment; }
    for(const p of pointLoads) if(x>=p.x-1e-8) {shear-=num(p.p); moment-=num(p.p)*(x-p.x);}
    for(const q of udls) {
      const a=clamp(Math.min(q.start,q.end),0,L), b=clamp(Math.max(q.start,q.end),0,L), z=Math.max(0,Math.min(x,b)-a);
      if(z>0){shear-=num(q.w)*z; moment-=num(q.w)*z*(x-(a+z/2));}
    }
    for(const q of triangles) {
      const a=clamp(Math.min(q.start,q.end),0,L), b=clamp(Math.max(q.start,q.end),0,L), z=Math.max(0,Math.min(x,b)-a);
      if(z>0 && b>a){
        const leftIsStart=q.start<=q.end, wa=leftIsStart?num(q.w1):num(q.w2), wb=leftIsStart?num(q.w2):num(q.w1);
        const wz=wa+(wb-wa)*z/(b-a), force=(wa+wz)*z/2;
        const centroid=a+z*(wa+2*wz)/(3*(wa+wz || 1));
        shear-=force; moment-=force*(x-centroid);
      }
    }
    return {x:Number(x.toFixed(3)),shear:Number(shear.toFixed(3)),moment:Number(moment.toFixed(3))};
  });
  const peak=data.reduce((best,p)=>Math.abs(p.moment)>Math.abs(best.moment)?p:best,data[0]);
  const maxShear=data.reduce((best,p)=>Math.abs(p.shear)>Math.abs(best.shear)?p:best,data[0]);
  const sumR=reactions.reduce((s,r)=>s+r.vertical,0);
  const sumLoads=pointLoads.reduce((s,p)=>s+num(p.p),0)
    +udls.reduce((s,q)=>s+num(q.w)*Math.abs(q.end-q.start),0)
    +triangles.reduce((s,q)=>s+(num(q.w1)+num(q.w2))*Math.abs(q.end-q.start)/2,0);
  return {data,reactions,peak,maxShear,equilibrium:sumR-sumLoads};
}

function Field({label,value,onChange,step='any',min,max}) {
  return <label className="field"><span>{label}</span><input type="number" step={step} min={min} max={max} value={value} onChange={e=>onChange(e.target.value)} /></label>;
}
function Panel({title,icon:Icon,accent,add,children}) {
  return <section className="panel"><div className="panel-head"><h2 style={{color:accent}}><Icon size={18}/>{title}</h2><button className="add" onClick={add}><Plus size={15}/> إضافة</button></div>{children}</section>;
}
function LoadRow({title,children,onRemove}) {
  return <div className="load-row"><div className="row-title"><b>{title}</b><button className="delete" onClick={onRemove} aria-label="حذف"><Trash2 size={16}/></button></div><div className="fields">{children}</div></div>;
}

export default function App() {
  const [L,setL]=useState(6);
  const [report,setReport]=useState(false);
  const [supports,setSupports]=useState([{id:1,x:0,type:'pin'},{id:2,x:6,type:'roller'}]);
  const [points,setPoints]=useState([]);
  const [udls,setUdls]=useState([{id:1,start:0,end:6,w:1}]);
  const [triangles,setTriangles]=useState([]);
  const [error,setError]=useState('');
  const result=useMemo(()=>{try{setError('');return analyze({L:num(L),supports,pointLoads:points,udls,triangles});}catch(e){return {error:e.message};}},[L,supports,points,udls,triangles]);

  const change=(setter,id,key,value)=>setter(old=>old.map(v=>v.id===id?{...v,[key]:key==='type'?value:num(value)}:v));
  const add=(setter,item)=>setter(old=>[...old,{id:uid(),...item}]);
  const remove=(setter,id)=>setter(old=>old.filter(v=>v.id!==id));
  const updateLength=v=>{const next=Math.max(1,num(v));setL(next);setSupports(s=>s.map(a=>({...a,x:Math.min(a.x,next)})));setUdls(s=>s.map(a=>({...a,end:Math.min(a.end,next)})));setTriangles(s=>s.map(a=>({...a,end:Math.min(a.end,next)})));setPoints(s=>s.map(a=>({...a,x:Math.min(a.x,next)})));};
  const reset=()=>{setL(6);setSupports([{id:1,x:0,type:'pin'},{id:2,x:6,type:'roller'}]);setPoints([]);setUdls([{id:1,start:0,end:6,w:1}]);setTriangles([]);};
  const chart=(key,color,title,unit)=> <section className="panel chart-panel"><h2 style={{color}}><Activity size={18}/>{title}</h2><div className="chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={result.data}><CartesianGrid strokeDasharray="3 3" stroke="#263449"/><XAxis dataKey="x" stroke="#94a3b8" tick={{fontSize:11}}/><YAxis stroke="#94a3b8" tick={{fontSize:11}}/><Tooltip contentStyle={{background:'#0f172a',border:'1px solid #334155',borderRadius:12,color:'#fff'}} formatter={(v)=>[`${v} ${unit}`,title]}/><ReferenceLine y={0} stroke="#64748b"/><Area type="monotone" dataKey={key} stroke={color} fill={color} fillOpacity={0.18} strokeWidth={2}/></AreaChart></ResponsiveContainer></div></section>;

  return <main className={report?'app report':'app'}>
    <header className="top panel"><div className="brand"><div className="logo"><BarChart3 size={26}/></div><div><h1>O9 <small>STRUCTURAL ANALYZER</small></h1><p>منظومة تحليل الأعتاب الهندسية</p></div></div><div className="top-actions"><label className="length">طول العتب (م)<input type="number" min="1" value={L} onChange={e=>updateLength(e.target.value)}/></label><button className="primary" onClick={()=>setReport(!report)}><Printer size={16}/>{report?'العودة للتعديل':'وضع الطباعة'}</button><button className="ghost" onClick={reset} title="إعادة ضبط"><RotateCcw size={17}/></button></div></header>
    {!report&&<aside className="notice"><AlertTriangle size={18}/><span>أداة مساعدة تعليمية أولية. تحقق من النموذج والأحمال والنتائج ببرنامج معتمد ومراجعة مهندس مختص قبل أي استخدام تنفيذي.</span></aside>}
    <div className="layout">
      {!report&&<div className="controls">
        <Panel title="المساند" icon={Layers} accent="#67e8f9" add={()=>add(setSupports,{x:Number((L/2).toFixed(2)),type:'roller'})}>
          {supports.map((s,i)=><LoadRow key={s.id} title={`مسند #${i+1}`} onRemove={()=>remove(setSupports,s.id)}><Field label="الموقع (م)" value={s.x} min="0" max={L} onChange={v=>change(setSupports,s.id,'x',clamp(v,0,L))}/><label className="field"><span>النوع</span><select value={s.type} onChange={e=>change(setSupports,s.id,'type',e.target.value)}><option value="pin">مفصلي</option><option value="roller">بكرة</option><option value="fixed">ثابت</option></select></label></LoadRow>)}
        </Panel>
        <Panel title="الأحمال النقطية" icon={Activity} accent="#f87171" add={()=>add(setPoints,{x:Number((L/2).toFixed(2)),p:10})}>
          {points.length===0&&<p className="empty">لا توجد أحمال نقطية</p>}
          {points.map((p,i)=><LoadRow key={p.id} title={`حمل نقطي #${i+1}`} onRemove={()=>remove(setPoints,p.id)}><Field label="الموقع (م)" value={p.x} min="0" max={L} onChange={v=>change(setPoints,p.id,'x',clamp(v,0,L))}/><Field label="P (kN)" value={p.p} onChange={v=>change(setPoints,p.id,'p',v)}/></LoadRow>)}
        </Panel>
        <Panel title="الأحمال المنتظمة (UDL)" icon={Activity} accent="#fbbf24" add={()=>add(setUdls,{start:0,end:num(L),w:1})}>
          {udls.map((q,i)=><LoadRow key={q.id} title={`حمل موزع #${i+1}`} onRemove={()=>remove(setUdls,q.id)}><Field label="w (kN/m)" value={q.w} onChange={v=>change(setUdls,q.id,'w',v)}/><Field label="من (م)" value={q.start} min="0" max={L} onChange={v=>change(setUdls,q.id,'start',clamp(v,0,L))}/><Field label="إلى (م)" value={q.end} min="0" max={L} onChange={v=>change(setUdls,q.id,'end',clamp(v,0,L))}/></LoadRow>)}
        </Panel>
        <Panel title="الأحمال المثلثية" icon={Activity} accent="#f0abfc" add={()=>add(setTriangles,{start:0,end:num(L),w1:0,w2:10})}>
          {triangles.length===0&&<p className="empty">لا توجد أحمال مثلثية</p>}
          {triangles.map((q,i)=><LoadRow key={q.id} title={`حمل متدرج #${i+1}`} onRemove={()=>remove(setTriangles,q.id)}><Field label="w1 (kN/m)" value={q.w1} onChange={v=>change(setTriangles,q.id,'w1',v)}/><Field label="w2 (kN/m)" value={q.w2} onChange={v=>change(setTriangles,q.id,'w2',v)}/><Field label="من (م)" value={q.start} min="0" max={L} onChange={v=>change(setTriangles,q.id,'start',clamp(v,0,L))}/><Field label="إلى (م)" value={q.end} min="0" max={L} onChange={v=>change(setTriangles,q.id,'end',clamp(v,0,L))}/></LoadRow>)}
        </Panel>
      </div>}
      <div className="results">
        {result.error?<section className="panel error">{result.error}</section>:<>
          <div className="stats">
            <div className="stat"><span>أقصى عزم انحناء |M|max</span><strong>{fmt(Math.abs(result.peak.moment))}<small> kN·m</small></strong><em>عند X = {fmt(result.peak.x)} م</em></div>
            <div className="stat"><span>أقصى قوة قص |V|max</span><strong>{fmt(Math.abs(result.maxShear.shear))}<small> kN</small></strong><em>عند X = {fmt(result.maxShear.x)} م</em></div>
            <div className="stat"><span>محصلة ردود الأفعال الرأسية</span><strong>{fmt(result.reactions.reduce((a,r)=>a+r.vertical,0))}<small> kN</small></strong><em>فحص اتزان رأسي</em></div>
          </div>
          <section className="panel"><h2><Layers size={18}/>ردود أفعال المساند</h2><div className="table-wrap"><table><thead><tr><th>المسند</th><th>الموقع (م)</th><th>رد الفعل الرأسي (kN)</th><th>عزم التثبيت (kN·m)</th></tr></thead><tbody>{result.reactions.map((r,i)=><tr key={r.id}><td>#{i+1}</td><td>{fmt(r.x)}</td><td>{fmt(r.vertical)}</td><td>{fmt(r.moment)}</td></tr>)}</tbody></table></div></section>
          <section className="panel"><h2><BarChart3 size={18}/>مخطط العتب والأحمال</h2><BeamDiagram L={num(L)} supports={supports} points={points} udls={udls} triangles={triangles}/></section>
          {chart('shear','#22d3ee','مخطط قوة القص (SFD)','kN')}
          {chart('moment','#c084fc','مخطط عزم الانحناء (BMD)','kN·m')}
          <section className="panel small-note"><b>ملاحظات النموذج:</b> يستخدم التحليل طريقة العناصر المحددة لعتبة Euler–Bernoulli مع صلابة انحناء EI ثابتة على طول العتب، وقيود رأسية للمساند المفصلية والبكرية، وقيد دوران إضافي للمسند الثابت. لا يتضمن حالياً الهبوط المسموح، القص المرن، التشوهات الكبيرة، أو تصميم التسليح.</section>
        </>}
      </div>
    </div>
    <footer>O9 Structural Analyzer · إصدار تجريبي 1.0</footer>
  </main>;
}

function BeamDiagram({L,supports,points,udls,triangles}) {
  const X=x=>40+520*x/L;
  return <div className="beam-wrap"><svg viewBox="0 0 600 190" role="img" aria-label="رسم العتب والمساند والأحمال">
    <line x1="40" y1="105" x2="560" y2="105" stroke="#67e8f9" strokeWidth="6" strokeLinecap="round"/>
    {udls.map(q=><g key={q.id}><rect x={X(Math.min(q.start,q.end))} y="68" width={Math.max(0,X(Math.max(q.start,q.end))-X(Math.min(q.start,q.end)))} height="37" fill="#fbbf2420" stroke="#fbbf24" strokeDasharray="4 3"/><text x={(X(q.start)+X(q.end))/2} y="60" fill="#fbbf24" textAnchor="middle" fontSize="11">{q.w} kN/m</text></g>)}
    {triangles.map(q=><g key={q.id}><polygon points={`${X(q.start)},105 ${X(q.start)},${105-Math.min(42,Math.abs(q.w1)*3)} ${X(q.end)},${105-Math.min(42,Math.abs(q.w2)*3)} ${X(q.end)},105`} fill="#e879f920" stroke="#e879f9"/><text x={(X(q.start)+X(q.end))/2} y="50" fill="#e879f9" textAnchor="middle" fontSize="10">{q.w1} → {q.w2}</text></g>)}
    {points.map(p=><g key={p.id}><line x1={X(p.x)} y1="30" x2={X(p.x)} y2="99" stroke="#f87171" strokeWidth="2"/><path d={`M ${X(p.x)-6} 91 L ${X(p.x)} 102 L ${X(p.x)+6} 91`} fill="#f87171"/><text x={X(p.x)} y="23" fill="#f87171" textAnchor="middle" fontSize="11">{p.p} kN</text></g>)}
    {supports.map(s=><g key={s.id} transform={`translate(${X(s.x)},105)`}>{s.type==='fixed'?<><rect x="-5" y="-22" width="10" height="44" fill="#c084fc"/><path d="M -13 -20 l8 -6 M -13 -10 l8 -6 M -13 0 l8 -6 M -13 10 l8 -6" stroke="#c084fc" strokeWidth="2"/></>:<><path d="M 0 0 L -13 20 L 13 20 Z" fill="#38bdf8"/>{s.type==='roller'&&<><circle cx="-7" cy="24" r="3" fill="#38bdf8"/><circle cx="7" cy="24" r="3" fill="#38bdf8"/></>}<line x1="-18" y1="29" x2="18" y2="29" stroke="#38bdf8" strokeWidth="2"/></>}</g>)}
    <text x="40" y="170" fill="#94a3b8" textAnchor="middle" fontSize="11">0 م</text><text x="560" y="170" fill="#94a3b8" textAnchor="middle" fontSize="11">{L} م</text>
  </svg></div>;
}
