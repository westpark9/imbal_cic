#!/usr/bin/env python3
"""Check the finished EXP59 page and its integrated tab in headless Chrome."""
import argparse
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPORT_ID = 'scorer_verifier_target_0928'

ERRORS = '''<script>window.__qaErrors=[];window.addEventListener('error',e=>window.__qaErrors.push(e.message));</script>'''
PAGE_CHECKS = '''
const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
const coverage=JSON.parse(document.getElementById('exp59-report-coverage').textContent);
check(coverage.cells>0&&document.querySelectorAll('.prob-cell').length===coverage.cells,'all completed predictors and classes represented');
check(document.querySelectorAll('.arm-panel').length===coverage.panels,'completed contexts represented');
check(document.querySelectorAll('.capability-panel').length===coverage.capability_experts,'all 8 experts have two capability views');
check([...document.querySelectorAll('table[data-kind$="-own-capability"] tbody tr,table[data-kind$="-near-capability"] tbody tr')].length===coverage.capability_class_rows,'all classes in both capability views');
for(const s of document.querySelectorAll('.capability-select')){
  const initial=s.value;
  const panels=[...document.querySelectorAll('.capability-panel')].filter(p=>p.dataset.dataset===s.dataset.dataset);
  for(const o of s.options){s.value=o.value;s.dispatchEvent(new Event('change'));
    check(panels.filter(p=>!p.hidden).length===1&&panels.find(p=>!p.hidden).dataset.expert===o.value,'expert '+s.dataset.dataset+' e'+o.value);
    check(document.documentElement.scrollWidth<=innerWidth+1,'expert panel fits viewport '+s.dataset.dataset+' e'+o.value);}
  s.value=initial;s.dispatchEvent(new Event('change'));
}
const metric=document.getElementById('quality-metric');
check(metric.value==='f1','F1 is the initial metric');
for(const m of ['precision','recall','ap','f1']){
  metric.value=m;metric.dispatchEvent(new Event('change'));
  check([...document.querySelectorAll('.prob-cell')].every(c=>c.textContent===Number(c.dataset[m]).toFixed(4)),'metric '+m);
}
for(const s of document.querySelectorAll('.arm-select,.class-select,.region-class-select')){
  const initial=s.value,arm=s.classList.contains('arm-select'),region=s.classList.contains('region-class-select'),kind=arm?'arm':'class';
  const panels=[...document.querySelectorAll('.'+(region?'region-class':kind)+'-panel')].filter(p=>p.dataset.dataset===s.dataset.dataset);
  for(const o of s.options){s.value=o.value;s.dispatchEvent(new Event('change'));
    check(panels.filter(p=>!p.hidden).length===1&&panels.find(p=>!p.hidden).dataset[kind]===o.value,kind+' '+s.dataset.dataset+' '+o.value);}
  s.value=initial;s.dispatchEvent(new Event('change'));
}
check(!/\\b(?:NaN|Infinity)\\b/.test([...document.querySelectorAll('td')].map(e=>e.textContent).join(' ')),'finite table values');
check(document.documentElement.scrollWidth<=innerWidth+1,'no page horizontal overflow');
check(document.querySelector('h1').textContent==='Expert 역량을 검증하는 평가','same 0918 title');
check(!document.getElementById('protocol'),'requested section 6 removed');
check(!document.querySelector('.equation'),'formulas removed');
check(!document.body.innerText.includes('R@0.1%'),'unrequested low-FPR metric removed');
check([...document.querySelectorAll('h2')].map(x=>x.id).join(',')==='global,oracle,composition,quality,context','requested five-section order');
'''
CAPABILITY_CHECKS = '''
const expectedCapability=EXPECTED_CAPABILITY;
for(const [ds,d] of Object.entries(expectedCapability.datasets)){
  for(const e of d.experts){
    for(const [scope,suffix] of [['residual','own'],['observable','near']]){
      const rows=[...document.querySelectorAll('table[data-kind="exp59-'+ds+'-e'+e.expert+'-'+suffix+'-capability"] tbody tr')];
      const source=e[scope].classes;
      check(rows.length===source.length,'class rows '+ds+' e'+e.expert+' '+scope);
      source.forEach((c,i)=>{
        const cells=[...rows[i].cells].map(x=>x.textContent.trim());
        check(cells[0]===c.name,'class order '+ds+' e'+e.expert+' '+scope+' '+c.name);
        const numeric=x=>Number(x.replaceAll(',',''));
        if(scope==='residual'){
          check(numeric(cells[1])===c.expert_metrics.support&&numeric(cells[4])===c.fixed&&numeric(cells[5])===c.harmed&&numeric(cells[6])===c.global_metrics.FP&&numeric(cells[7])===c.expert_metrics.FP,'own counts '+ds+' e'+e.expert+' '+c.name);
        }else{
          const counts=cells[1].split('/').map(numeric),fp=cells[5].split('→').map(numeric);
          check(counts[0]===c.expert_metrics.support&&counts[1]===c.negative_rows&&fp[0]===c.global_metrics.FP&&fp[1]===c.expert_metrics.FP,'near counts '+ds+' e'+e.expert+' '+c.name);
          for(const [key,col] of [['precision',2],['recall',3],['f1',4]]){
            const fmt=m=>(key==='precision'?!m.predicted:!m.support)?'—':m[key].toFixed(4);
            check(cells[col]===fmt(c.global_metrics)+' → '+fmt(c.expert_metrics),'near '+key+' '+ds+' e'+e.expert+' '+c.name);
          }
        }
      });
    }
  }
}
'''
INDEX_CHECKS = '''
const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
const button=document.querySelector('[data-id="scorer_verifier_target_0928"]');
check(!!button,'new navigation tab');button.click();
check(document.getElementById('viewer').dataset.loaded==='scorer_verifier_target_0928','new tab loaded');
check(document.getElementById('viewer').src.startsWith('data:text/html;base64,'),'embedded report loaded');
check(document.getElementById('tb-time').textContent==='2026-09-28','new tab date');
check(button.querySelector('.title-kr').textContent===document.querySelector('[data-id="scorer_verifier_target_0918"] .title-kr').textContent,'same navigation title');
check(!!document.querySelector('[data-id="scorer_verifier_target_0918"]'),'old tab retained');
check(document.documentElement.scrollWidth<=innerWidth+1,'no page horizontal overflow');
'''


def browser_check(source, checks, out, name, size):
    chrome=shutil.which('google-chrome') or shutil.which('chromium')
    if not chrome:raise RuntimeError('Chrome unavailable')
    with tempfile.TemporaryDirectory(prefix='exp59-html-qa-') as tmp:
        tmp=Path(tmp)
        finish='''<script>window.addEventListener('load',()=>setTimeout(()=>{
          let result;try{CHECKS
            if(window.__qaErrors.length)throw Error(window.__qaErrors.join('; '));
            result={ok:true,checks,width:innerWidth};
          }catch(e){result={ok:false,error:String(e),errors:window.__qaErrors};}
          const el=document.createElement('pre');el.id='exp59-qa-result';el.hidden=true;
          el.textContent=JSON.stringify(result);document.body.appendChild(el);
        },500));</script>'''.replace('CHECKS',checks)
        page=source.read_text().replace('<head>','<head>'+ERRORS,1).replace('</body>',finish+'</body>',1)
        target=tmp/'check.html';target.write_text(page)
        cmd=[chrome,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
             '--no-first-run','--no-default-browser-check',f'--user-data-dir={tmp / "profile"}',
             '--hide-scrollbars',f'--window-size={size}','--virtual-time-budget=3000',
             f'--screenshot={out / (name+".png")}','--dump-dom',target.as_uri()]
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        (out/(name+'.browser.log')).write_text(proc.stderr)
        if proc.returncode:raise RuntimeError(f'Chrome exited {proc.returncode}: {proc.stderr[-1500:]}')
        match=re.search(r'<pre id="exp59-qa-result"[^>]*>(.*?)</pre>',proc.stdout,re.S)
        if not match:raise RuntimeError(f'Browser checks did not run: {name}')
        result=json.loads(html.unescape(match.group(1)))
        (out/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        if not result['ok']:raise AssertionError(result)
        return result


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    page=ROOT/'lablog/html_report'/f'{REPORT_ID}.html'
    index=ROOT/'lablog/html_report/post_tabpfn.html'
    capability=json.loads((ROOT/'tabpfn/results/20260928_exp59_residual_oracle_s43/expert_capability/capability.json').read_text())
    page_checks=PAGE_CHECKS+CAPABILITY_CHECKS.replace('EXPECTED_CAPABILITY',json.dumps(capability,ensure_ascii=False))
    result={}
    for name,source,checks,size in [
        ('report_desktop',page,page_checks,'1440,1100'),
        ('report_mobile',page,page_checks,'390,844'),
        ('integrated_desktop',index,INDEX_CHECKS,'1440,1100')]:
        result[name]=browser_check(source,checks,out,name,size)
    (out/'COMPLETE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v['ok'] for k,v in result.items()}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'tabpfn/results/20260928_exp59_residual_oracle_s43/report_qa')
    run(p.parse_args().out)
