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
check(document.querySelectorAll('.prob-cell').length===255,'all 13 predictors and all classes represented');
check(document.querySelectorAll('.arm-panel').length===6,'three contexts for each dataset');
const metric=document.getElementById('quality-metric');
for(const m of ['r','f1','ap']){
  metric.value=m;metric.dispatchEvent(new Event('change'));
  check([...document.querySelectorAll('.prob-cell')].every(c=>c.textContent===Number(c.dataset[m]).toFixed(4)),'metric '+m);
}
for(const s of document.querySelectorAll('.arm-select,.class-select')){
  const initial=s.value,arm=s.classList.contains('arm-select'),kind=arm?'arm':'class';
  const panels=[...document.querySelectorAll('.'+kind+'-panel')].filter(p=>p.dataset.dataset===s.dataset.dataset);
  for(const o of s.options){s.value=o.value;s.dispatchEvent(new Event('change'));
    check(panels.filter(p=>!p.hidden).length===1&&panels.find(p=>!p.hidden).dataset[kind]===o.value,kind+' '+s.dataset.dataset+' '+o.value);}
  s.value=initial;s.dispatchEvent(new Event('change'));
}
check(!/\\b(?:NaN|Infinity)\\b/.test([...document.querySelectorAll('td')].map(e=>e.textContent).join(' ')),'finite table values');
check(document.documentElement.scrollWidth<=innerWidth+1,'no page horizontal overflow');
check(document.querySelector('h1').textContent==='Expert 역량을 검증하는 평가','same 0918 title');
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
    result={}
    for name,source,checks,size in [
        ('report_desktop',page,PAGE_CHECKS,'1440,1100'),
        ('report_mobile',page,PAGE_CHECKS,'390,844'),
        ('integrated_desktop',index,INDEX_CHECKS,'1440,1100')]:
        result[name]=browser_check(source,checks,out,name,size)
    (out/'COMPLETE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v['ok'] for k,v in result.items()}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'tabpfn/results/20260928_exp59_residual_oracle_s43/report_qa')
    run(p.parse_args().out)
