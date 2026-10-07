#!/usr/bin/env python3
"""Render the local EXP60 report and verify every expert/view against its data."""
import json
from pathlib import Path

from check_exp59_report import browser_check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/research/20260929/exp60_html_qa'
REPORT = ROOT / 'lablog/html_report/counterexample_context_0929.html'
INDEX = REPORT.with_name('post_tabpfn.html')

PAGE_CHECKS = r'''
const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
const data=JSON.parse(document.getElementById('exp60-data').textContent);
const set=(id,value)=>{const el=document.getElementById(id);el.value=value;el.dispatchEvent(new Event('change'));};
const rows=kind=>[...document.querySelectorAll('table[data-kind="'+kind+'"] tbody tr')].map(r=>[...r.cells].map(c=>c.textContent.trim()));
const num=x=>Number(x.replaceAll(',',''));
const pair=x=>x.split('/').map(num);
check(Object.keys(data.datasets).length===2,'both datasets present');
check(data.seed===43&&data.K===4,'single seed and K shown');
for(const scope of ['input_nearest','residual_oracle']){
 set('overall-scope',scope);
 for(const [ds,d] of Object.entries(data.datasets)){
  const models=['global',...data.arms.map(a=>a+'_'+scope)], rendered=rows(ds+'-overall');
  check(rendered.length===4,'four conditions '+ds+' '+scope);
  models.forEach((model,i)=>{
   const s=d.summary[model],r=rendered[i];
   check(r[1]===s.macro_f1.toFixed(4)&&r[2]===(s.accuracy*100).toFixed(2)+'%','whole-set metrics '+ds+' '+model);
   check(num(r[3])===s.fixed&&num(r[4])===s.harmed,'whole-set correction counts '+ds+' '+model);
  });
 }
 for(const metric of ['precision','recall','f1']){
  set('overall-metric',metric);
  for(const [ds,d] of Object.entries(data.datasets))check(rows(ds+'-overall-classes').length===d.names.length,'whole-set classes '+ds+' '+scope+' '+metric);
 }
}
for(const scope of ['residual','observable']){
 set('summary-scope',scope);
 for(const [ds,d] of Object.entries(data.datasets)){
  const r=rows(ds+'-expert-summary');
  check(r.length===4,'four experts '+ds+' '+scope);
  Object.entries(d.experts).forEach(([k,e],i)=>{
   check(num(r[i][1])===e[scope].designed.rows,'evaluation rows '+ds+' e'+k+' '+scope);
   data.arms.forEach((arm,j)=>{const p=pair(r[i][j+2]),s=e[scope][arm];check(p[0]===s.fixed&&p[1]===s.harmed,'summary corrections '+ds+' e'+k+' '+scope+' '+arm);});
  });
 }
}
for(const [ds,d] of Object.entries(data.datasets)){
 set('detail-dataset',ds);
 for(const [k,e] of Object.entries(d.experts)){
  set('detail-expert',k);
  for(const scope of ['residual','observable']){
   set('detail-scope',scope);
   const source=e[scope], counts=rows('detail-counts'),totals=rows('detail-totals');
   check(counts.length===d.names.length+1,'all class counts plus total '+ds+' e'+k+' '+scope);
   data.arms.forEach((arm,j)=>{
    const value=source[arm],top=totals[j],end=pair(counts.at(-1)[4+j]);
    check(num(top[1])===value.fixed&&num(top[2])===value.harmed&&num(top[3])===value.expert_correct,'detail totals '+ds+' e'+k+' '+scope+' '+arm);
    check(end[0]===value.fixed&&end[1]===value.harmed,'total row '+ds+' e'+k+' '+scope+' '+arm);
    let fixed=0,harmed=0;
    value.classes.forEach((c,i)=>{
     const p=pair(counts[i][1+j]),q=pair(counts[i][4+j]);
     check(counts[i][0]===c.name&&p[0]===c.expert_metrics.TP&&p[1]===c.expert_metrics.FP&&q[0]===c.fixed&&q[1]===c.harmed,'class counts '+ds+' e'+k+' '+scope+' '+arm+' '+c.name);
     fixed+=q[0];harmed+=q[1];
    });
    check(fixed===value.fixed&&harmed===value.harmed,'visible class sum equals headline '+ds+' e'+k+' '+scope+' '+arm);
   });
   for(const metric of ['precision','recall','f1']){
    set('detail-metric',metric);
    const metricRows=rows('detail-metrics');
    check(metricRows.length===d.names.length,'metric class coverage '+ds+' e'+k+' '+scope+' '+metric);
    const expected=m=>((metric==='precision'&&m.TP+m.FP===0)||(metric!=='precision'&&m.support===0))?'—':m[metric].toFixed(4);
    source.designed.classes.forEach((c,i)=>{
     check(num(metricRows[i][1])===c.expert_metrics.support&&metricRows[i][2]===expected(c.global_metrics),'global reference '+ds+' e'+k+' '+scope+' '+metric+' '+c.name);
     data.arms.forEach((arm,j)=>check(metricRows[i][3+j]===expected(source[arm].classes[i].expert_metrics),'expert metric '+ds+' e'+k+' '+scope+' '+metric+' '+arm+' '+c.name));
    });
    check(document.documentElement.scrollWidth<=innerWidth+1,'detail fits viewport '+ds+' e'+k+' '+scope+' '+metric);
   }
  }
 }
}
set('overall-scope','input_nearest');set('overall-metric','f1');set('summary-scope','observable');
set('detail-dataset','cic2018');set('detail-expert','3');set('detail-scope','observable');set('detail-metric','f1');
check(!/\b(?:NaN|Infinity)\b/.test([...document.querySelectorAll('td')].map(x=>x.textContent).join(' ')),'finite rendered table values');
check(document.documentElement.scrollWidth<=innerWidth+1,'no page horizontal overflow');
'''

INDEX_CHECKS = r'''
const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
const tab=document.querySelector('[data-id="counterexample_context_0929"]');
check(!!tab,'0929 navigation tab');tab.click();
const frame=document.getElementById('viewer');
check(frame.dataset.loaded==='counterexample_context_0929','0929 report selected');
check(frame.src.startsWith('data:text/html;base64,'),'self-contained report loaded');
check(document.getElementById('tb-time').textContent==='2026-09-29','0929 report date');
check(!!document.querySelector('[data-id="scorer_verifier_target_0918"]'),'0918 tab preserved');
check(!!document.querySelector('[data-id="scorer_verifier_target_0928"]'),'0928 tab preserved');
check(document.documentElement.scrollWidth<=innerWidth+1,'index fits viewport');
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result={}
    for name,source,checks,size in [
        ('report_desktop',REPORT,PAGE_CHECKS,'1440,1100'),
        ('report_mobile',REPORT,PAGE_CHECKS,'390,844'),
        ('expert_detail',REPORT,PAGE_CHECKS+"for(const el of document.querySelector('main').children){if(el.id!=='detail'&&!el.classList.contains('viewer'))el.hidden=true;}",'1440,1100'),
        ('integrated_desktop',INDEX,INDEX_CHECKS,'1440,1100')]:
        result[name]=browser_check(source,checks,OUT,name,size)
    (OUT/'COMPLETE.json').write_text(json.dumps({k:{'ok':v['ok'],'checks':len(v['checks']),'width':v['width']} for k,v in result.items()},ensure_ascii=False,indent=2)+'\n')
    print((OUT/'COMPLETE.json').read_text())


if __name__=='__main__':
    main()
