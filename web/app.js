const SUGGEST=["How many books can I borrow from the library?","What happens if I miss the MSE?","Is there a dress code on exam days?","What fee concessions and scholarships are available?","What are the hostel rules?","What is the vision and mission of CHRIST?"];
const $=id=>document.getElementById(id);let busy=false;const hist=[];
const esc=s=>s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function md(s){s=esc(s).replace(/\*\*(.+?)\*\*/g,"<b>$1</b>").replace(/\[(\d+)\]/g,'<sup class="cite">[$1]</sup>');
 return s.split("\n").map(l=>/^\s*[-•*] /.test(l)?'<div class="li">• '+l.replace(/^\s*[-•*] /,"")+"</div>":l).join("\n").replace(/<\/div>\n/g,"</div>")}
function showChips(){$("chips").innerHTML=SUGGEST.map(q=>`<button data-q="${q}">${q}</button>`).join("")}
const AVATAR={u:"/static/photo.png",b:"/static/photo2.png"};
function add(role,html){const row=document.createElement("div");row.className="msg "+role;
 const av=document.createElement("img");av.className="avatar";av.src=AVATAR[role];av.alt=role==="u"?"You":"Helpdesk bot";av.onerror=()=>{av.style.display="none"};
 const d=document.createElement("div");d.className="m "+role;d.innerHTML=html;row.append(av,d);$("msgs").append(row);$("msgs").scrollTop=1e9;return d}
function extras(r){const f=document.createDocumentFragment();
 if(r.sources&&r.sources.length){const s=document.createElement("details");s.className="src";
  s.innerHTML="<summary>📖 Sources ("+r.sources.length+")</summary>"+r.sources.map(x=>`<div><span class="tag">[${x.n}]</span><b>${esc(x.section)}</b> · p.${x.page}<br>${esc(x.snippet)}</div>`).join("");f.append(s)}
 if(r.id&&r.model!=="guardrail"){const b=document.createElement("div");b.className="fb";b.innerHTML=`Helpful? <button data-v="up">👍</button><button data-v="down">👎</button><span style="margin-left:auto">${r.latency_ms} ms</span>`;
  b.onclick=e=>{const v=e.target.dataset.v;if(!v)return;fetch("/api/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:r.id,vote:v})});b.querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===e.target))};f.append(b)}
 return f}
async function ask(q){q=q.trim();if(!q||busy)return;busy=true;const w=$("welcome");if(w)w.remove();add("u",esc(q));
 const t=add("b",'<span class="dots"><span></span><span></span><span></span></span>');
 try{const r=await (await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:q,history:hist.slice(-6),model:curModel})})).json();
  t.innerHTML=md(r.answer);if(r.model&&r.model!=="guardrail"&&r.model!=="none"){const g=document.createElement("div");g.className="mtag";g.textContent="model: "+r.model;t.append(g)}t.append(extras(r));hist.push({role:"user",content:q},{role:"assistant",content:r.answer})}catch(e){t.textContent="Sorry, the helpdesk is unreachable right now. Please try again."}
 busy=false;$("msgs").scrollTop=1e9;$("input").focus()}
document.addEventListener("click",e=>{const c=e.target.closest("[data-q]");if(c)ask(c.dataset.q)});
$("form").onsubmit=e=>{e.preventDefault();const v=$("input").value;$("input").value="";ask(v)};
$("clear").onclick=()=>{location.reload()};
showChips();


/* ---- model picker (cards in the sidebar) ---- */
let curModel=null;
function renderModels(list){$("models").innerHTML=list.length?list.map(m=>`<button class="mcard ${m.id===curModel?"on":""}" data-m="${m.id}"><span class="radio"></span><span><div class="mname">${esc(m.label.replace(" (local)",""))}</div><div class="mmeta">${m.params} params${m.size_mb?" · "+m.size_mb+" MB":""} · on this PC</div></span></button>`).join(""):'<p class="hint">No local model downloaded yet - using offline quote mode.</p>';
 $("modelNote").textContent=curModel?"Runs locally on your CPU: private, no internet needed. Larger models answer better but take longer.":""}
let modelList=[];
function loadModels(){fetch("/api/models").then(r=>r.json()).then(d=>{modelList=d.models;if(!curModel||!modelList.some(m=>m.id===curModel))curModel=d.default;renderModels(modelList)}).catch(()=>{})}
$("models").onclick=e=>{const c=e.target.closest("[data-m]");if(c){curModel=c.dataset.m;renderModels(modelList)}};
loadModels();
/* ---- tabs ---- */
document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("on",x===b));
 $("tab-chat").hidden=b.dataset.tab!=="chat";$("tab-compare").hidden=b.dataset.tab!=="compare";if(b.dataset.tab==="compare")loadCompare();else loadModels()});
/* ---- comparison tab ---- */
const pct=x=>x==null?"–":Math.round(x*100)+"%";
const sec=x=>x==null?"–":x.toFixed(1)+" s";
const num=x=>x==null?"–":Math.round(x).toLocaleString();
const dec1=x=>x==null?"–":x.toFixed(1);
const has=(m,k)=>m[k]!==undefined&&m[k]!==null;
const overall=m=>{const v=["fact_accuracy","faithfulness","citation_rate","abstain_rate"].filter(k=>has(m,k)).map(k=>m[k]);return v.reduce((a,b)=>a+b,0)/v.length};
const SECTIONS=[
 {t:"Answer quality",cols:[["fact_accuracy","Fact accuracy",pct,1],["exact_rate","Fully correct",pct,1],["faithfulness","Faithfulness",pct,1],["relevancy","Answer relevancy",pct,1],["citation_rate","Cites source",pct,1]]},
 {t:"Safety & honesty",cols:[["abstain_rate","Admits “not in handbook”",pct,1],["hallucination_rate","Hallucination on trick questions",pct,-1],["bad_number_rate","Answers with invented numbers",pct,-1]]},
 {t:"Efficiency & token usage",cols:[["avg_latency_s","Avg time",sec,-1],["tokens_per_sec","Speed (tokens/s)",dec1,1],["avg_prompt_tokens","Prompt tokens / answer",num,-1],["avg_completion_tokens","Output tokens / answer",num,0],["avg_total_tokens","Total tokens / answer",num,-1],["avg_answer_words","Answer length (words)",num,0]]}];
function bars(title,list,key,fmt,gold){const v=list.filter(e=>has(e.metrics,key));if(!v.length)return"";const mx=Math.max(...v.map(e=>e.metrics[key]),1e-9);
 return `<div class="chart"><h3>${title}</h3>${v.map(e=>`<div class="row"><span>${esc(e.label)}</span><div class="track"><div class="fill ${gold?"gold":""}" style="width:${Math.max(3,e.metrics[key]/mx*100)}%"></div></div><span>${fmt(e.metrics[key])}</span></div>`).join("")}</div>`}
function loadCompare(){fetch("/api/compare").then(r=>r.json()).then(d=>{const L=d.models||[],box=$("cmp");
 if(!L.length){box.innerHTML='<h2>Model Comparison</h2><p class="sub">No results yet. Run <code>python -m eval.compare_models</code> in the project folder, then reopen this tab.</p>';return}
 const vals=k=>L.filter(e=>has(e.metrics,k));
 const best=(k,hi=true)=>{const v=vals(k);return v.length?v.reduce((a,b)=>(hi?b.metrics[k]>a.metrics[k]:b.metrics[k]<a.metrics[k])?b:a):null};
 const bo=L.reduce((a,b)=>overall(b.metrics)>overall(a.metrics)?b:a);
 const win=(t,e,v)=>e?`<div class="win"><small>${t}</small><b>${esc(e.label)}</b><span>${v}</span></div>`:"";
 const star=(e,k,dir)=>dir!==0&&L.length>1&&has(e.metrics,k)&&best(k,dir>0)&&best(k,dir>0).id===e.id&&vals(k).some(o=>o.metrics[k]!==e.metrics[k])?"best":"";
 const ba=best("fact_accuracy"),bf=best("faithfulness"),bh=best("abstain_rate"),bs=best("avg_latency_s",false),bt=best("avg_total_tokens",false);
 const first=L[0],tbl=s=>`<h3 class="sech">${s.t}</h3><table><tr><th>Model</th>${s.cols.map(c=>`<th>${c[1]}</th>`).join("")}</tr>${L.map(e=>`<tr><td><b>${esc(e.label)}</b></td>${s.cols.map(c=>`<td class="${star(e,c[0],c[3])}">${c[2](has(e.metrics,c[0])?e.metrics[c[0]]:null)}</td>`).join("")}</tr>`).join("")}</table>`;
 const rc=vals("context_recall")[0];
 box.innerHTML=`<h2>Model Comparison</h2>
 <p class="sub">Same ${first.n_answerable} handbook questions + ${first.n_unanswerable} trick questions (answer not in the handbook), given to every model with identical retrieved context and prompt (${esc(first.prompt_version)}). Local models are 4-bit quantised and run on this PC's CPU. Last run: ${esc(first.ran_at)}.</p>
 <div class="wins">${win("Best overall",bo,pct(overall(bo.metrics)))}${win("Most accurate",ba,pct(ba&&ba.metrics.fact_accuracy))}${bf?win("Most faithful",bf,pct(bf.metrics.faithfulness)):""}${win("Most honest",bh,pct(bh&&bh.metrics.abstain_rate))}${win("Fastest",bs,sec(bs&&bs.metrics.avg_latency_s))}${bt?win("Fewest tokens",bt,num(bt.metrics.avg_total_tokens)+" / answer"):""}</div>
 <table><tr><th>Model</th><th>Size</th><th>Parameters</th><th>Overall</th></tr>${L.slice().sort((a,b)=>overall(b.metrics)-overall(a.metrics)).map(e=>`<tr><td><b>${esc(e.label)}</b></td><td>${e.size_mb?e.size_mb+" MB":"cloud"}</td><td>${esc(e.params||"")}</td><td class="${bo.id===e.id&&L.length>1?"best":""}">${pct(overall(e.metrics))}</td></tr>`).join("")}</table>
 ${SECTIONS.map(tbl).join("")}
 ${rc?`<p class="sub"><b>Retrieval (identical for every model):</b> context recall ${pct(rc.metrics.context_recall)}, context precision ${pct(rc.metrics.context_precision)}. High recall means the handbook passages given to the model contained the answer, so any mistakes come from the model, not the search.</p>`:""}
 ${bars("Faithfulness: answer is supported by the handbook text (higher is better)",L,"faithfulness",pct)}
 ${bars("Fact accuracy",L,"fact_accuracy",pct)}
 ${bars("Hallucination on trick questions (lower is better)",L,"hallucination_rate",pct,true)}
 ${bars("Answers with invented numbers (lower is better)",L,"bad_number_rate",pct,true)}
 ${bars("Total tokens per answer (lower is cheaper)",L,"avg_total_tokens",num,true)}
 ${bars("Generation speed in tokens per second (higher is better)",L,"tokens_per_sec",dec1)}
 ${bars("Average response time (lower is better)",L,"avg_latency_s",sec,true)}
 <details class="det"><summary>What do these metrics mean?</summary><div class="q">
 <b>Overall</b> = average of fact accuracy, faithfulness, citation rate and honesty.
 <b>Fact accuracy</b>: share of the expected facts (numbers, names, dates) that appear in the answer.
 <b>Faithfulness</b>: share of the answer's statements that are backed by the retrieved handbook text (semantic match).
 <b>Answer relevancy</b>: how close the answer is in meaning to the question.
 <b>Cites source</b>: answer includes a [n] reference to a handbook passage.
 <b>Admits “not in handbook”</b>: on trick questions the model correctly says it cannot find the answer; <b>Hallucination</b> is the opposite (it answered anyway).
 <b>Invented numbers</b>: the answer contains a number that does not appear in the retrieved text.
 <b>Context recall / precision</b>: how many needed facts the retrieved passages contain, and how many passages were useful.
 <b>Tokens</b>: prompt = question + handbook excerpts fed to the model, output = the generated answer. Local models cost nothing per token but more tokens means slower answers.</div></details>
 ${L.map(e=>`<details class="det"><summary>${esc(e.label)}: every answer (${e.details.filter(x=>x.ok).length}/${e.details.length} correct)</summary>${e.details.map(x=>`<div class="q"><span class="${x.ok?"ok":"bad"}">${x.ok?"✔":"✘"}</span> <b>${esc(x.q)}</b>${x.kind==="unanswerable"?" <i>(trick question)</i>":""}<br>${md(x.answer||"(no answer / error)")}<br><span class="note">${x.latency_s}s${x.prompt_tokens?" · "+x.prompt_tokens+" in / "+x.completion_tokens+" out tokens":""}${x.faithfulness!=null?" · faithfulness "+pct(x.faithfulness):""}${x.bad_number?" · ⚠ number not in context":""}</span></div>`).join("")}</details>`).join("")}`}).catch(()=>{$("cmp").innerHTML='<p class="sub">Could not load results.</p>'})}
