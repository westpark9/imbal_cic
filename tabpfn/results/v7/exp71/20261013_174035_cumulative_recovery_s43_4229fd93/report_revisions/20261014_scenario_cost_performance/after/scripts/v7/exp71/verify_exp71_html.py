#!/usr/bin/env python3
"""Exercise every EXP71 report control in Chrome and capture desktop/mobile views."""
import argparse
import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

try:
    import websocket
except ImportError:
    sys.path.insert(0,'/tmp/exp61-browser-tools')
    import websocket

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'lablog/html_report').is_dir())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--qa-dir',type=Path)
    a=parser.parse_args();dest=a.qa_dir.resolve() if a.qa_dir else a.run.resolve()/'qa';dest.mkdir(parents=True,exist_ok=True)
    chrome=shutil.which('google-chrome') or shutil.which('chromium')
    if not chrome:raise RuntimeError('Chrome unavailable')
    errors=[];counter=0
    with tempfile.TemporaryDirectory(prefix='exp71_browser_') as profile:
        with (dest/'chrome.log').open('w') as log:
            proc=subprocess.Popen([chrome,'--headless=new','--disable-gpu','--no-sandbox',
                '--no-first-run','--remote-debugging-port=0','--remote-allow-origins=*',
                '--user-data-dir='+profile,'about:blank'],stdout=log,stderr=subprocess.STDOUT)
            try:
                port_file=Path(profile)/'DevToolsActivePort'
                for _ in range(100):
                    if port_file.exists():break
                    if proc.poll() is not None:raise RuntimeError('Chrome failed')
                    time.sleep(.1)
                port=int(port_file.read_text().splitlines()[0])
                pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
                target=next(p for p in pages if p['type']=='page')
                ws=websocket.create_connection(target['webSocketDebuggerUrl'],timeout=20,suppress_origin=True)
                def call(method,params=None):
                    nonlocal counter
                    counter+=1;ident=counter;ws.send(json.dumps(dict(id=ident,method=method,params=params or {})))
                    while True:
                        msg=json.loads(ws.recv())
                        if msg.get('method')=='Runtime.exceptionThrown':errors.append(msg)
                        if msg.get('id')==ident:
                            if 'error' in msg:raise RuntimeError(msg['error'])
                            return msg.get('result',{})
                def evaluate(code):
                    r=call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True))
                    if r.get('exceptionDetails'):raise RuntimeError(r['exceptionDetails'])
                    return r['result'].get('value')
                def screenshot(name):
                    data=call('Page.captureScreenshot',{'format':'png'})['data']
                    (dest/name).write_bytes(base64.b64decode(data))
                call('Page.enable');call('Runtime.enable')
                call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1000,deviceScaleFactor=1,mobile=False))
                uri=(ROOT/'lablog/html_report/bounded_updates_1013.html').as_uri()
                call('Page.navigate',{'url':uri})
                for _ in range(100):
                    if evaluate("document.getElementById('class-table')?.tBodies[0].rows.length > 0"):break
                    time.sleep(.1)
                result=evaluate(r'''(()=>{
                  const checked=[];
                  for(const policy of ['bounded','cumulative']) for(const ds of ['cic2018','toniot']) for(const c of ['uniform','natural']){
                    el('dataset').value=ds;el('case').value=c;el('policy').value=policy;resetStages();
                    const rr=records(ds,c,'tabpfn');
                    for(const r of rr){
                      el('stage-select').value=r.stage;render();
                      const expected=[...r.classes].sort((a,b)=>b.support-a.support);
                      const actual=[...el('class-table').tBodies[0].rows];
                      if(actual.length!==expected.length)throw Error('class row count');
                      actual.forEach((row,i)=>{
                        const v=[...row.cells].map(x=>x.textContent);
                        const p=expected[i],x=records(ds,c,'xgb')[r.stage].classes.find(q=>q.name===p.name);
                        const want=[N[p.name],num(p.support),num(r.incoming_class_counts[p.name]),num(r.fit_class_counts[p.name]),pct(x.precision),pct(x.recall),pct(x.f1),pct(p.precision),pct(p.recall),pct(p.f1)];
                        want.forEach((w,j)=>{if(v[j]!==w)throw Error('rendered metric mismatch '+[policy,ds,c,r.stage,j]);});
                        for(let j=1;j<10;j++)if(getComputedStyle(row.cells[j]).textAlign!=='right')throw Error('numeric alignment');
                      });
                      if(el('stages').tBodies[0].rows.length!==rr.length)throw Error('stage row count');
                      const x=el('chart').querySelector('[data-model="xgb"]'),p=el('chart').querySelector('[data-model="tabpfn"]');
                      if(x.getAttribute('stroke')!=='#D55E00'||x.getAttribute('stroke-dasharray')!=='8 5'||p.getAttribute('stroke')!=='#0072B2'||p.hasAttribute('stroke-dasharray'))throw Error('model color / dash mismatch');
                      checked.push([policy,ds,c,r.stage]);
                    }
                  }
                  const summaries=summaryRows();
                  if(el('cost-summary').tBodies[0].rows.length!==8||el('performance-summary').tBodies[0].rows.length!==8)throw Error('summary count');
                  summaries.forEach((s,i)=>{
                    const x=records(s.dataset,s.case,'xgb',s.policy).at(-1),p=records(s.dataset,s.case,'tabpfn',s.policy).at(-1);
                    const v=[...el('cost-summary').tBodies[0].rows[i].cells].map(x=>x.textContent);
                    const want=[label(s),num(p.context_rows),dec(s.xgb.update_sum),dec(s.tabpfn.update_sum),dec(x.predict_seconds),dec(p.predict_seconds),num(p.rows)];
                    if(JSON.stringify(v)!==JSON.stringify(want))throw Error('cost scope / summary mismatch');
                    const f=[...el('performance-summary').tBodies[0].rows[i].cells].map(x=>x.textContent);
                    if(f[1]!==pct(s.xgb.final_macro_f1)||f[2]!==pct(s.tabpfn.final_macro_f1)||f[4]!==rate(s.xgb.final_benign_fpr)||f[5]!==rate(s.tabpfn.final_benign_fpr))throw Error('performance summary mismatch');
                  });
                  const sectionOrder=[...document.querySelectorAll('main > section')].map(x=>x.id);
                  if(sectionOrder.join(',')!=='scenario,cost,performance,conclusion')throw Error('narrative order');
                  const ids=[...document.querySelectorAll('[id]')].map(x=>x.id);
                  if(new Set(ids).size!==ids.length)throw Error('duplicate HTML IDs');
                  if(/undefined|NaN/.test(document.body.innerText))throw Error('invalid visible value');
                  el('dataset').value='cic2018';el('case').value='uniform';el('policy').value='bounded';resetStages();
                  return {states:checked.length,summary_rows:8,sectionOrder,checked};
                })()''')
                if evaluate('Boolean(DATA.cumulative)'):
                    comparison=evaluate(r'''(()=>{
                      const checked=[];
                      for(const ds of ['cic2018','toniot']) for(const c of ['uniform','natural']) for(const m of ['xgb','tabpfn']){
                        el('dataset').value=ds;el('case').value=c;el('policy').value='cumulative';el('history-method').value=m;resetStages();
                        const rr=records(ds,c,m,'cumulative'),br=records(ds,c,m,'bounded');
                        for(const r of rr){
                          el('stage-select').value=r.stage;render();
                          const expected=[...r.classes].sort((a,b)=>b.support-a.support);
                          const actual=[...el('history-classes').tBodies[0].rows];
                          if(actual.length!==expected.length)throw Error('comparison class row count');
                          actual.forEach((row,i)=>{
                            const v=[...row.cells].map(x=>x.textContent),q=expected[i];
                            if(v[0]!==N[q.name]||v[1]!==num(q.support)||v[2]!==num(br[r.stage].fit_class_counts[q.name])||v[3]!==num(r.fit_class_counts[q.name])||v[9]!==pct(q.f1))throw Error('comparison metric mismatch');
                          });
                          checked.push([ds,c,m,r.stage]);
                        }
                      }
                      if(/undefined|NaN/.test(document.body.innerText))throw Error('comparison invalid value');
                      el('dataset').value='toniot';el('case').value='natural';el('policy').value='cumulative';el('history-method').value='tabpfn';resetStages();
                      return {states:checked.length,summary_rows:8,checked};
                    })()''')
                    result['comparison']=comparison
                    evaluate("el('history-classes').closest('details').open=true;el('history-classes').scrollIntoView();window.scrollBy(0,-100)")
                    screenshot('desktop-cumulative-classes.png')
                    evaluate("el('history-classes').closest('details').open=false;el('dataset').value='cic2018';el('case').value='uniform';el('policy').value='bounded';resetStages();el('cost').scrollIntoView();window.scrollBy(0,-90)")
                    screenshot('desktop-cost.png')
                    evaluate("el('performance').scrollIntoView();window.scrollBy(0,-90)")
                    screenshot('desktop-performance.png')
                    evaluate("el('conclusion').scrollIntoView();window.scrollBy(0,-90)")
                    screenshot('desktop-conclusion.png')
                    evaluate('window.scrollTo(0,0)')
                screenshot('desktop-summary.png')
                evaluate("el('classes').scrollIntoView();window.scrollBy(0,-80)")
                screenshot('desktop-classes.png')
                call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
                evaluate('window.scrollTo(0,0)');time.sleep(.2);screenshot('mobile-summary.png')
                mobile=evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
                assert mobile['scroll']<=mobile['width']+1,mobile
                # Verify the new tab and its embedded source in the actual index.
                call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1000,deviceScaleFactor=1,mobile=False))
                call('Page.navigate',{'url':(ROOT/'lablog/html_report/post_tabpfn.html').as_uri()+'#bounded_updates_1013'})
                for _ in range(100):
                    if evaluate("document.querySelector('.nav-item.active')?.dataset.id === 'bounded_updates_1013'"):break
                    time.sleep(.1)
                integrated=evaluate("({active:document.querySelector('.nav-item.active')?.dataset.id,title:document.getElementById('tb-title')?.textContent,frames:[...document.querySelectorAll('iframe')].map(f=>({id:f.id,hasReport:f.src.startsWith('data:text/html;base64,')&&atob(f.src.split(',')[1]).includes('EXP71')}))})")
                assert integrated['active']=='bounded_updates_1013',integrated
                assert any(f['hasReport'] for f in integrated['frames']),integrated
                time.sleep(.7)
                screenshot('integrated-tab.png');assert not errors,errors
                result.update(status='passed',console_errors=errors,mobile=mobile,integrated=integrated)
                (dest/'html_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
                print(json.dumps(result,ensure_ascii=False,indent=2));ws.close()
            finally:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':main()
