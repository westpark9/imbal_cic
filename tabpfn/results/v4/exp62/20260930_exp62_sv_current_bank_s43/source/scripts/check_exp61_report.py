#!/usr/bin/env python3
"""Check every SOTA metric/control and the 0908 embedded report in Chrome."""
import json
from pathlib import Path
from check_exp59_report import browser_check

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/research/20260930/exp61_html_qa'
REPORT=ROOT/'lablog/html_report/sota_pfn_comparison.html'

PAGE=r'''
const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
const data=JSON.parse(document.getElementById('exp61-data').textContent);
const rows=kind=>[...document.querySelectorAll('table[data-kind="'+kind+'"] tbody tr')].map(r=>[...r.cells].map(c=>c.textContent.trim()));
const num=s=>Number(s.replaceAll(',',''));
const set=(id,val)=>{const el=document.getElementById(id);el.value=val;el.dispatchEvent(new Event('change'));};
check(data.seed===43,'single seed 43');
check(!document.body.innerText.includes('상대사례'),'withdrawn experiment absent');
const section=document.getElementById('exp61-results');
check(section.previousElementSibling.querySelector('table').textContent.includes('BoostPFN'),'results immediately after paper introduction');
check(document.getElementById('historical-sota')&&!document.getElementById('historical-sota').open,'old split results folded');
const summary=rows('exp61-summary');
check(summary.length===6,'six model rows');
for(let i=0;i<data.methods.length;i++){
 const m=data.methods[i],a=data.datasets.cic2018.results[m],b=data.datasets.toniot.results[m],r=summary[i];
 check(r[1]===a.macro_f1.toFixed(4)&&r[3]===b.macro_f1.toFixed(4),'macro values '+m);
 check(r[2]===(a.accuracy*100).toFixed(2)+'%'&&r[4]===(b.accuracy*100).toFixed(2)+'%','accuracy '+m);
}
for(const metric of ['f1','precision','recall']){
 set('exp61-metric',metric);
 for(const [ds,d] of Object.entries(data.datasets)){
  const table=rows('exp61-'+ds+'-classes');check(table.length===d.class_order.length,'all classes '+ds+' '+metric);
  d.class_order.forEach((name,i)=>{
   check(table[i][0]===data.class_names[name],'class name '+ds+' '+name);
   check(num(table[i][1])===d.results.global_raw.classes[i].support,'support '+ds+' '+name);
   data.methods.forEach((m,j)=>check(table[i][j+2]===d.results[m].classes[i][metric].toFixed(4),'metric '+ds+' '+m+' '+name+' '+metric));
  });
 }
}
for(const [ds,d] of Object.entries(data.datasets)){
 for(const m of data.methods){
  set('exp61-'+ds+'-method',m);
  const panels=[...document.querySelectorAll('.exp61-method-panel')].filter(p=>p.dataset.dataset===ds&&!p.hidden);
  check(panels.length===1&&panels[0].dataset.method===m,'method selection '+ds+' '+m);
  const table=rows('exp61-'+ds+'-'+m+'-counts');
  d.results[m].classes.forEach((c,i)=>{
   check(num(table[i][1])===c.support,'detail support '+ds+' '+m+' '+c.name);
   ['precision','recall','f1'].forEach((key,j)=>check(table[i][j+2]===c[key].toFixed(4),'detail rate '+ds+' '+m+' '+c.name+' '+key));
   ['TP','FP','FN'].forEach((key,j)=>check(num(table[i][j+5])===c[key],'detail count '+ds+' '+m+' '+c.name+' '+key));
  });
 }
 const cost=rows('exp61-'+ds+'-cost');
 data.methods.forEach((m,i)=>{
  const r=d.results[m];
  ['fit_seconds','validation_seconds_included_in_fit','predict_seconds','seconds_per_1000_rows','peak_gpu_sampled_gib','peak_model_rss_sampled_gib'].forEach((key,j)=>check(num(cost[i][j+1])===Number(r[key].toFixed(j===3?4:2)),'cost '+ds+' '+m+' '+key));
 });
 set('exp61-'+ds+'-method','global_raw');
}
set('exp61-metric','f1');
check(document.documentElement.scrollWidth<=innerWidth+1,'no page horizontal overflow');
check(!/\b(?:NaN|Infinity)\b/.test([...document.querySelectorAll('#exp61-results td')].map(e=>e.textContent).join(' ')),'finite table values');
document.getElementById('exp61-title').scrollIntoView();
'''

INDEX=r'''
const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
check(!document.querySelector('[data-id="counterexample_context_0929"]'),'withdrawn tab removed');
check(!document.body.innerText.includes('상대사례'),'withdrawn overview text removed');
const tab=document.querySelector('[data-id="sota_pfn_comparison"]');check(!!tab,'0908 navigation retained');tab.click();
check(document.getElementById('tb-time').textContent==='2026-09-08','0908 date retained');
const frame=document.getElementById('viewer');check(frame.dataset.loaded==='sota_pfn_comparison','0908 selected');
const body=atob(frame.src.split(',')[1]);
check(body.includes('exp61-results')&&body.includes('exp61-data'),'latest data embedded');
check(document.querySelectorAll('[data-id]').length===12,'11 reports plus overview');
check(document.documentElement.scrollWidth<=innerWidth+1,'index fits viewport');
'''

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reports={}
    for name,source,checks,size in [('report_desktop',REPORT,PAGE,'1440,1100'),('report_mobile',REPORT,PAGE,'390,844'),('integrated_desktop',REPORT.with_name('post_tabpfn.html'),INDEX,'1440,1100'),('class_tables',REPORT,PAGE+"document.querySelector('table[data-kind=\"exp61-cic2018-classes\"]').scrollIntoView();",'1440,1100')]:
        reports[name]=browser_check(source,checks,OUT,name,size)
    (OUT/'COMPLETE.json').write_text(json.dumps({k:dict(ok=v['ok'],checks=len(v['checks'])) for k,v in reports.items()},indent=2)+'\n')
    print((OUT/'COMPLETE.json').read_text())

if __name__=='__main__':main()
