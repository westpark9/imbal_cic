#!/usr/bin/env python3
"""Verify merged navigation, all report values, and expert controls in Chrome."""
import base64
import json
from pathlib import Path
import re

from check_exp59_report import browser_check

ROOT=Path(__file__).resolve().parents[1]
HTML=ROOT/'lablog/html_report'
DOC=ROOT/'docs/research/20260930'
OUT=DOC/'exp63_capability_html_qa'
REPORT=HTML/'scorer_verifier_target_0928.html'
PAGE=r'''
const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
const data=JSON.parse(document.getElementById('merged-data').textContent);
const rows=kind=>[...document.querySelectorAll('table[data-kind="'+kind+'"] tbody tr')].map(r=>[...r.cells].map(c=>c.textContent.trim()));
const num=s=>Number(s.replaceAll(',',''));
check([...document.querySelectorAll('section>h2')].length===4,'four main sections');
check([...document.querySelectorAll('section')].map(s=>s.id).join(',')==='current,oracle,capability,sv','narrative order');
check(!document.querySelector('.capability-panel'),'old A/B capability panels replaced');
check(!document.body.innerText.includes('상대사례'),'withdrawn intervention absent');
check(!document.body.innerText.includes('R@0.1%'),'unrequested metric absent');
check(!document.querySelector('.equation'),'no equations');
check(data.seed===43&&data.K===4,'current seed and K');
const summary=rows('current-summary');let i=0;
for(const [ds,d] of Object.entries(data.datasets))for(const arm of Object.keys(data.arms)){
 const s=d.summaries[arm],r=summary[i++];
 check(r[0]===d.title&&r[1]===data.arms[arm],'summary identity '+ds+' '+arm);
 check(r[2]===s.macro_f1.toFixed(4)&&r[3]===(100*s.accuracy).toFixed(2)+'%','summary metrics '+ds+' '+arm);
 check(num(r[5])===s.helpful&&num(r[6])===s.harmful,'summary corrections '+ds+' '+arm);
}
const metric=document.getElementById('current-metric');
for(const key of ['system_f1','precision','recall']){
 metric.value=key;metric.dispatchEvent(new Event('change'));
 for(const [ds,d] of Object.entries(data.datasets)){
  const table=rows('current-'+ds+'-classes');
  check(table.length===d.class_order.length,'all classes '+ds+' '+key);
  d.class_order.forEach((cl,i)=>{
   check(table[i][0]===data.class_names[cl]&&num(table[i][1])===d.classes.global[i].support,'class identity '+ds+' '+cl);
   Object.keys(data.arms).forEach((arm,j)=>check(table[i][j+2]===d.classes[arm][i][key].toFixed(4),'class metric '+ds+' '+arm+' '+cl+' '+key));
  });
 }
}
metric.value='system_f1';metric.dispatchEvent(new Event('change'));
const activation=rows('sv-activation');
Object.entries(data.datasets).forEach(([ds,d],i)=>{
 const s=d.summaries.s1v0,r=activation[i];
 check(num(r[1])===s.proposed&&num(r[3])===s.accepted&&num(r[5])===s.changed&&num(r[6])===s.helpful&&num(r[7])===s.harmful,'S/V counts '+ds);
 check(r[2]===(100*s.proposed/s.rows).toFixed(2)+'%'&&r[4]===(100*s.accepted/s.rows).toFixed(2)+'%','S/V rates '+ds);
 const o=rows('diagnostic-oracle')[i];check(o[1]===d.diagnostic_global.toFixed(4)&&o[2]===d.oracle.toFixed(4),'oracle '+ds);
 const changes=rows('sv-'+ds+'-changes');
 d.classes.s1v0.forEach((c,j)=>check(changes[j][3]===c.system_f1.toFixed(4)&&num(changes[j][4])===c.helpful&&num(changes[j][5])===c.harmful,'class corrections '+ds+' '+c.class));
});
check(data.datasets.cic2018.summaries.s1v0.proposed===0,'CIC fallback');
check(data.datasets.toniot.summaries.s1v0.proposed===data.datasets.toniot.summaries.s1v0.rows,'ToN full call');
check(!document.getElementById('history').open,'historical experiments folded');
check([...document.querySelectorAll('[data-report-jump]')].some(x=>x.dataset.reportJump==='sota_pfn_comparison'),'SOTA navigation');
check(!/\b(?:NaN|Infinity)\b/.test([...document.querySelectorAll('td')].map(e=>e.textContent).join(' ')),'finite displayed values');
check(document.documentElement.scrollWidth<=innerWidth+1,'no page overflow');
'''
SWEEP=r'''
const sw=data.capability_sweep;
const change=(id,value)=>{const e=document.getElementById('sweep-'+id);e.value=String(value);e.dispatchEvent(new Event('change'));};
check(sw.seed===43&&sw.ks.join(',')==='2,4,6,8','sweep seed and K');
check(document.getElementById('assignment-correction').textContent.includes('이전 해석은 철회'),'A/B interpretation corrected');
let overviewIndex=0;
for(const [ds,d] of Object.entries(sw.datasets)){
 change('dataset',ds);
 for(const [k,b] of Object.entries(d.banks)){
  change('k',k);const overview=rows('sweep-overview')[overviewIndex++];
  check(overview[0]===d.title&&overview[1]===k,'K summary identity '+ds+k);
  check(overview[2]===d.targets.map(c=>data.class_names[c]+' '+b.both_improved[c]+'/'+k).join(' / '),'same expert paired improvement '+ds+k);
  check(overview[3]===b.single_class_blocks+'/'+k&&num(overview[4])===b.total_context_rows,'context totals '+ds+k);
  check(overview[5]===b.policy.macro_f1.toFixed(4)&&overview[6]===(100*b.policy.proposed/b.policy.rows).toFixed(0)+'%','K policy results '+ds+k);
  for(const split of ['full_test','cal_confirm'])for(const metric of ['f1','precision','recall']){
   change('split',split);change('metric',metric);
   check(rows('sweep-matrix').length===b.K+1,'all experts matrix '+ds+k+split+metric);
   for(const [model,m] of Object.entries(b.models[split]))m.classes.forEach((c,i)=>{
    const cell=document.querySelector('#sweep-matrix td[data-model="'+model+'"][data-class="'+c.class+'"]');
    check(cell.querySelector('span').textContent===c[metric].toFixed(4),'matrix absolute '+ds+k+split+metric+model+c.class);
    const delta=c[metric]-b.models[split].global.classes[i][metric];
    check(cell.querySelector('small').textContent===(model==='global'?'기준':(delta>=0?'+':'')+delta.toFixed(4)),'matrix change '+ds+k+split+metric+model+c.class);
   });
   check(document.documentElement.scrollWidth<=innerWidth+1,'matrix fits viewport '+ds+k+split+metric);
  }
  for(let e=1;e<=b.K;e++){
   change('expert',e);const m=b.models.full_test['expert'+e],v=b.models.cal_confirm['expert'+e],g=b.models.full_test.global,vg=b.models.cal_confirm.global,ctx=b.contexts[e-1];
   const paired=rows('sweep-paired'),counts=rows('sweep-counts');
   check(paired.length===d.names.length&&counts.length===d.names.length,'all detail classes '+ds+k+e);
   check(document.getElementById('sweep-detail-title').textContent.includes('K='+k+' · e'+e),'detail identity '+ds+k+e);
   const tc=[...document.querySelectorAll('table[data-kind="sweep-paired"] tbody tr')];
   d.names.forEach((c,i)=>{
    const a=m.classes[i],base=g.classes[i],z=v.classes[i];
    check(tc[i].cells[2].querySelector('span').textContent===base.f1.toFixed(4)+' → '+a.f1.toFixed(4),'paired test F1 '+ds+k+e+c);
    check(tc[i].cells[3].querySelector('span').textContent===vg.classes[i].f1.toFixed(4)+' → '+z.f1.toFixed(4),'paired confirm F1 '+ds+k+e+c);
    check(paired[i][4]===a.precision.toFixed(4)&&paired[i][5]===a.recall.toFixed(4),'detail P R '+ds+k+e+c);
    check(paired[i][6].split(' → ').map(num).join(',')===[base.FP,a.FP].join(','),'detail FP '+ds+k+e+c);
    check(tc[i].cells[1].childNodes[0].textContent===ctx.anchor[c].toLocaleString('en-US')+' + '+ctx.block[c].toLocaleString('en-US'),'anchor block '+ds+k+e+c);
    check(num(counts[i][1])===a.support&&num(counts[i][5])===a.helpful&&num(counts[i][6])===a.harmful,'counts support corrections '+ds+k+e+c);
    for(const [key,column] of [['TP',2],['FP',3],['FN',4]])check(counts[i][column].split(' → ').map(num).join(',')===[base[key],a[key]].join(','),'detail '+key+ds+k+e+c);
    const badge=tc[i].cells[0].querySelector('.sweep-tag');check(Boolean(badge)===(a.delta_f1>1e-12&&z.delta_f1>1e-12),'paired improvement badge '+ds+k+e+c);
   });
   const selected=document.querySelector('#sweep-matrix button[aria-pressed="true"]');check(selected&&selected.dataset.expert===String(e),'selected expert highlighted '+ds+k+e);
   document.getElementById('sweep-count-details').open=true;
   check(document.documentElement.scrollWidth<=innerWidth+1,'expanded detail fits viewport '+ds+k+e);
   document.getElementById('sweep-count-details').open=false;
  }
  document.querySelector('#sweep-matrix button[data-expert="1"]').click();check(document.getElementById('sweep-expert').value==='1','matrix button selects detail '+ds+k);
 }
}
change('dataset','cic2018');change('k',4);change('split','full_test');change('metric','f1');
'''
INDEX=r'''
const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
const id='scorer_verifier_target_0928';
check(document.querySelectorAll('.nav-item').length===10,'one merged tab replaces three');
check(!document.querySelector('[data-id="clean_revalidation_0916"]')&&!document.querySelector('[data-id="scorer_verifier_target_0918"]'),'old tabs removed from navigation');
check(document.querySelectorAll('[data-id="'+id+'"]').length===1,'one canonical tab');
check(!!document.querySelector('[data-id="dataset_quality_0911"]')&&!!document.querySelector('[data-id="sota_pfn_comparison"]'),'audit and SOTA tabs retained');
for(const alias of ['clean_revalidation_0916','scorer_verifier_target_0918',id]){
 activate(alias);check(document.getElementById('viewer').dataset.loaded===id,'old hash alias '+alias);
 check(location.hash==='#'+id,'canonical hash '+alias);
}
check(document.getElementById('tb-time').textContent==='2026-09-30','latest report date');
const frame=document.getElementById('viewer');
check(atob(frame.src.split(',')[1]).includes('merged-data'),'new report embedded');
window.dispatchEvent(new MessageEvent('message',{source:frame.contentWindow,data:{type:'lablog:navigate',reportId:'sota_pfn_comparison'}}));
check(frame.dataset.loaded==='sota_pfn_comparison','iframe button navigates to SOTA');
check(atob(frame.src.split(',')[1]).includes('exp61-results'),'SOTA results preserved');
activate(id);
check(!document.body.innerText.includes('상대사례'),'withdrawn overview absent');
check(document.documentElement.scrollWidth<=innerWidth+1,'index fits viewport');
'''

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=json.loads((DOC/'merged_expert_report_data.json').read_text())
    checks=PAGE+SWEEP
    results={}
    for name,source,code,size in [
        ('report_desktop',REPORT,checks,'1440,1100'),('report_mobile',REPORT,checks,'390,844'),
        ('integrated_desktop',HTML/'post_tabpfn.html',INDEX,'1440,1100'),
        ('integrated_mobile',HTML/'post_tabpfn.html',INDEX,'390,844'),
        ('expert_detail',REPORT,checks+"document.getElementById('sweep-detail').scrollIntoView();",'1440,1100'),
        ('sv_section',REPORT,checks+"document.getElementById('sv').scrollIntoView();",'1440,1100')]:
        result=browser_check(source,code,OUT,name,size);results[name]={'ok':result['ok'],'checks':len(result['checks'])}
    def embedded(path):return {r['id']:r for r in json.loads(re.search(r'const REPORTS = (.*?);\n',path.read_text()).group(1))}
    new=embedded(HTML/'post_tabpfn.html')
    # Other tabs may be rebuilt by their own builders (EXP61, report_order); they must match the files on disk, checked below.
    for key in ['clean_revalidation_0916','scorer_verifier_target_0918']:
        text=(HTML/(key+'.html')).read_text();assert 'http-equiv="refresh"' in text and 'post_tabpfn.html#scorer_verifier_target_0928' in text
    for r in new.values():
        if r['type']=='iframe':assert base64.b64decode(r['b64'])==(HTML/(r['id']+'.html')).read_bytes()
    results['source_checks']={'other_tabs_match_disk':True,'legacy_files_redirect':True,'all_embedded_sources_match':True}
    (OUT/'COMPLETE.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

if __name__=='__main__':main()
