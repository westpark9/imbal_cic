#!/usr/bin/env python3
"""Verify merged navigation, all report values, and expert controls in Chrome."""
import base64
import json
from pathlib import Path
import re

from check_exp59_report import browser_check, CAPABILITY_CHECKS

ROOT=Path(__file__).resolve().parents[1]
HTML=ROOT/'lablog/html_report'
DOC=ROOT/'docs/research/20260930'
OUT=DOC/'merged_expert_html_qa'
REPORT=HTML/'scorer_verifier_target_0928.html'
PAGE=r'''
const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
const data=JSON.parse(document.getElementById('merged-data').textContent);
const rows=kind=>[...document.querySelectorAll('table[data-kind="'+kind+'"] tbody tr')].map(r=>[...r.cells].map(c=>c.textContent.trim()));
const num=s=>Number(s.replaceAll(',',''));
check([...document.querySelectorAll('section>h2')].length===4,'four main sections');
check([...document.querySelectorAll('section')].map(s=>s.id).join(',')==='current,oracle,capability,sv','narrative order');
check(document.querySelectorAll('.capability-panel').length===8,'eight experts preserved');
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
for(const s of document.querySelectorAll('.capability-select')){
 const original=s.value;const parent=document.getElementById('experts-'+s.dataset.dataset);parent.open=true;
 for(const option of s.options){s.value=option.value;s.dispatchEvent(new Event('change'));
  const panels=[...document.querySelectorAll('.capability-panel')].filter(p=>p.dataset.dataset===s.dataset.dataset&&!p.hidden);
  check(panels.length===1&&panels[0].dataset.expert===option.value,'expert selector '+s.dataset.dataset+' '+option.value);
  check(document.documentElement.scrollWidth<=innerWidth+1,'expert fits viewport '+s.dataset.dataset+' '+option.value);
 }
 s.value=original;s.dispatchEvent(new Event('change'));parent.open=false;
}
check(!document.getElementById('history').open,'historical experiments folded');
check([...document.querySelectorAll('[data-report-jump]')].some(x=>x.dataset.reportJump==='sota_pfn_comparison'),'SOTA navigation');
check(!/\b(?:NaN|Infinity)\b/.test([...document.querySelectorAll('td')].map(e=>e.textContent).join(' ')),'finite displayed values');
check(document.documentElement.scrollWidth<=innerWidth+1,'no page overflow');
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
    checks=PAGE+CAPABILITY_CHECKS.replace('EXPECTED_CAPABILITY',json.dumps(data['capability'],ensure_ascii=False))
    results={}
    for name,source,code,size in [
        ('report_desktop',REPORT,checks,'1440,1100'),('report_mobile',REPORT,checks,'390,844'),
        ('integrated_desktop',HTML/'post_tabpfn.html',INDEX,'1440,1100'),
        ('integrated_mobile',HTML/'post_tabpfn.html',INDEX,'390,844'),
        ('expert_detail',REPORT,checks+"document.getElementById('experts-cic2018').open=true;document.getElementById('experts-cic2018').scrollIntoView();",'1440,1100'),
        ('sv_section',REPORT,checks+"document.getElementById('sv').scrollIntoView();",'1440,1100')]:
        result=browser_check(source,code,OUT,name,size);results[name]={'ok':result['ok'],'checks':len(result['checks'])}
    def embedded(path):return {r['id']:r for r in json.loads(re.search(r'const REPORTS = (.*?);\n',path.read_text()).group(1))}
    old=embedded(DOC/'archive/pre_merge_reports/post_tabpfn.html');new=embedded(HTML/'post_tabpfn.html')
    for key in new:
        if key not in ['scorer_verifier_target_0928','overview']:assert old[key]==new[key],f'Unrelated tab modified: {key}'
    for key in ['clean_revalidation_0916','scorer_verifier_target_0918']:
        text=(HTML/(key+'.html')).read_text();assert 'http-equiv="refresh"' in text and 'post_tabpfn.html#scorer_verifier_target_0928' in text
    for r in new.values():
        if r['type']=='iframe':assert base64.b64decode(r['b64'])==(HTML/(r['id']+'.html')).read_bytes()
    results['source_checks']={'other_tabs_unchanged':True,'legacy_files_redirect':True,'all_embedded_sources_match':True}
    (OUT/'COMPLETE.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

if __name__=='__main__':main()
