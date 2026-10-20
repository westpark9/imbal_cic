#!/usr/bin/env python3
"""Check report filters, source-derived values, ordering and embedded-tab rendering."""
from pathlib import Path
import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

_root = next(p for p in Path(__file__).resolve().parents
             if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import repo_root
ROOT = repo_root(__file__)
try:
    import websocket
except ImportError:
    sys.path.insert(0, '/tmp/exp61-browser-tools')
    import websocket

CHECK = r'''(()=>{
 const checks=[], stages=[];
 const texts=row=>[...row.cells].map(x=>x.textContent);
 for(const ds of Object.keys(D))for(const c of Object.keys(C))for(const p of Object.keys(P)){
  el('dataset').value=ds;el('case').value=c;el('policy').value=p;render();
  const rows=[...el('overall').querySelectorAll('tbody tr')];
  if(rows.length!==4)throw Error('overall rows');
  rows.forEach((row,i)=>{
   const d=Object.keys(D)[Math.floor(i/2)],a=Object.keys(C)[i%2];
   methods.forEach((m,k)=>{if(texts(row)[k+1]!==pct(result(d,a,m,p).macro_f1))throw Error('overall value');});
  });
  for(const id of ['classwise','paired-classes','class-pr']){
   for(const table of el(id).querySelectorAll('table')){
    const rs=[...table.tBodies[0].rows],supports=rs.map(r=>Number(r.cells[1].textContent.replaceAll(',','')));
    if(supports.some((v,i)=>i&&v>supports[i-1]))throw Error('class order');
    rs.forEach(r=>{for(let j=1;j<r.cells.length;j++)if(getComputedStyle(r.cells[j]).textAlign!=='right')throw Error('numeric alignment');});
   }
  }
  const ours=result(ds,c,'ours',p),rs=[...el('paired-classes').querySelectorAll('tbody tr')];
  const sum=k=>rs.reduce((a,r)=>a+Number(r.cells[k].textContent.replaceAll(',','')),0);
  if(sum(8)!==ours.policy.helpful||sum(9)!==ours.policy.harmful)throw Error('paired class correction totals');
  const source=[...ours.classes].sort((a,b)=>b.support-a.support);
  rs.forEach((row,i)=>{if(texts(row)[7]!==pct(source[i].f1))throw Error('paired class F1');});
  const costs=[...el('costs').querySelectorAll('tbody tr')];
  costs.forEach((row,i)=>{
   const j=job(ds,c,methods[i]),v=texts(row);
   if(v[2]!==num(j.updates.slice(1).reduce((a,b)=>a+b,0),2))throw Error('initial excluded from updates');
   if(v[4]!==num(j.results[p].predict_service_seconds,2))throw Error('policy-specific inference');
  });
  for(const m of ['xgb','tabpfn']){
   const last=el('fit-'+m).querySelector('tbody').lastElementChild;
   if(Number(last.cells[3].textContent.replaceAll(',',''))<8000||last.cells[4].textContent!=='0')throw Error('supplemental validation access versus use');
  }
  for(const opt of el('stage').options){
   el('stage').value=opt.value;renderStage();
   const s=Number(opt.value),base=result(ds,c,'xgb',p).stages[s];
   const expected=[...base.classes].filter(x=>x.support>0).sort((a,b)=>b.support-a.support);
   const actual=[...el('stage-classes').querySelectorAll('tbody tr')];
   if(expected.length!==actual.length)throw Error('unseen class rows');
   actual.forEach((row,i)=>{
    if(texts(row)[0]!==N[expected[i].name])throw Error('stage class order');
    methods.forEach((m,k)=>{
     const v=result(ds,c,m,p).stages[s].classes.find(x=>x.name===expected[i].name);
     if(texts(row)[k+2]!==pct(v.f1))throw Error('stage class F1');
    });
   });
   stages.push([ds,c,p,s]);
  }
  if(el('performance-chart').querySelectorAll('.bar-fill').length!==6)throw Error('bar count');
  if(/NaN|undefined/.test(document.body.innerText))throw Error('invalid display');
  checks.push([ds,c,p]);
 }
 const ids=[...document.querySelectorAll('[id]')].map(x=>x.id);
 if(ids.length!==new Set(ids).size)throw Error('duplicate IDs');
 for(const a of document.querySelectorAll('a[href^="#"]'))if(!document.getElementById(a.getAttribute('href').slice(1)))throw Error('broken anchor');
 if(el('stage-details').open||el('class-details').open)throw Error('details should start collapsed');
 if(document.body.innerText.includes('후속 비교 · 이번 큐에서 제외'))throw Error('stale SOTA status');
 el('dataset').value='cic2018';el('case').value='uniform';el('policy').value='rate_1000';render();
 return {conditions:checks.length,stage_conditions:stages.length,checks,stages};
})()'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qa-dir', type=Path, required=True)
    dest = p.parse_args().qa_dir.resolve(); dest.mkdir(parents=True, exist_ok=True)
    errors = []; counter = 0
    with tempfile.TemporaryDirectory(prefix='exp73_report_') as profile:
        with (dest / 'chrome.log').open('w') as log:
            proc = subprocess.Popen([shutil.which('google-chrome'), '--headless=new', '--disable-gpu',
                '--no-sandbox', '--no-first-run', '--remote-debugging-port=0', '--remote-allow-origins=*',
                '--user-data-dir=' + profile, 'about:blank'], stdout=log, stderr=subprocess.STDOUT)
            try:
                port_file = Path(profile) / 'DevToolsActivePort'
                for _ in range(100):
                    if port_file.exists(): break
                    if proc.poll() is not None: raise RuntimeError('Chrome failed')
                    time.sleep(.1)
                port = int(port_file.read_text().splitlines()[0])
                pages = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
                target = next(x for x in pages if x['type'] == 'page')
                ws = websocket.create_connection(target['webSocketDebuggerUrl'], timeout=30, suppress_origin=True)
                def call(method, params=None):
                    nonlocal counter
                    counter += 1; ident = counter
                    ws.send(json.dumps(dict(id=ident, method=method, params=params or {})))
                    while True:
                        msg = json.loads(ws.recv())
                        if msg.get('method') == 'Runtime.exceptionThrown': errors.append(msg)
                        if msg.get('id') == ident:
                            if 'error' in msg: raise RuntimeError(msg['error'])
                            return msg.get('result', {})
                def evaluate(code):
                    r = call('Runtime.evaluate', dict(expression=code, returnByValue=True, awaitPromise=True))
                    if r.get('exceptionDetails'): raise RuntimeError(r['exceptionDetails'])
                    return r['result'].get('value')
                def screenshot(name):
                    (dest / name).write_bytes(base64.b64decode(call('Page.captureScreenshot', {'format': 'png'})['data']))
                call('Page.enable'); call('Runtime.enable')
                call('Emulation.setDeviceMetricsOverride', dict(width=1440, height=1000, deviceScaleFactor=1, mobile=False))
                call('Page.navigate', {'url': (ROOT / 'lablog/html_report/sequential_replay_1014.html').as_uri()})
                for _ in range(100):
                    if evaluate("document.querySelectorAll('#overall tbody tr').length === 4"): break
                    time.sleep(.1)
                result = evaluate(CHECK)
                screenshot('desktop-header.png')
                evaluate("el('results').scrollIntoView();window.scrollBy(0,-15)")
                screenshot('desktop-results.png')
                evaluate("el('dataset').value='toniot';render();el('policy-summary').scrollIntoView();window.scrollBy(0,-65)")
                screenshot('desktop-sv.png')
                evaluate("el('costs').scrollIntoView();window.scrollBy(0,-150)")
                screenshot('desktop-costs.png')
                evaluate("el('class-details').open=true;el('class-details').scrollIntoView();window.scrollBy(0,-20)")
                screenshot('desktop-classes.png')
                call('Emulation.setDeviceMetricsOverride', dict(width=390, height=844, deviceScaleFactor=1, mobile=True))
                evaluate('window.scrollTo(0,0)'); screenshot('mobile-header.png')
                evaluate("el('results').scrollIntoView()")
                screenshot('mobile-results.png')
                mobile = evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
                assert mobile['scroll'] <= mobile['width'] + 1, mobile
                call('Emulation.setDeviceMetricsOverride', dict(width=1440, height=1000, deviceScaleFactor=1, mobile=False))
                call('Page.navigate', {'url': (ROOT / 'lablog/html_report/post_tabpfn.html').as_uri() + '#sequential_replay_1014'})
                for _ in range(100):
                    if evaluate("document.querySelector('.nav-item.active')?.dataset.id === 'sequential_replay_1014'"): break
                    time.sleep(.1)
                integrated = evaluate("({active:document.querySelector('.nav-item.active')?.dataset.id,title:document.getElementById('tb-title')?.textContent,loaded:[...document.querySelectorAll('iframe')].some(f=>f.src.startsWith('data:text/html;base64,')&&atob(f.src.split(',')[1]).includes('exp73-data'))})")
                assert integrated['active'] == 'sequential_replay_1014' and integrated['loaded'], integrated
                time.sleep(.3); screenshot('integrated-tab.png')
                assert not errors, errors
                result.update(status='passed', console_errors=errors, mobile=mobile, integrated=integrated)
                (dest / 'html_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
                print(json.dumps({k: v for k, v in result.items() if k not in ['checks', 'stages']}, ensure_ascii=False, indent=2))
                ws.close()
            finally:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()


if __name__ == '__main__':
    main()
