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
    a=parser.parse_args();dest=a.run.resolve()/'qa';dest.mkdir(exist_ok=True)
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
                  for(const ds of ['cic2018','toniot']) for(const c of ['uniform','natural']){
                    el('dataset').value=ds;el('case').value=c;resetStages();
                    const rr=records(ds,c,'tabpfn');
                    for(const r of rr){
                      el('stage-select').value=r.stage;render();
                      const expected=[...r.classes].sort((a,b)=>b.support-a.support);
                      const actual=[...el('class-table').tBodies[0].rows];
                      if(actual.length!==expected.length)throw Error('class row count');
                      actual.forEach((row,i)=>{
                        const v=[...row.cells].map(x=>x.textContent);
                        if(v[0]!==N[expected[i].name]||v[1]!==num(expected[i].support)||v[9]!==pct(expected[i].f1))throw Error('rendered metric mismatch');
                      });
                      if(el('stages').tBodies[0].rows.length!==rr.length)throw Error('stage row count');
                      checked.push([ds,c,r.stage]);
                    }
                  }
                  const ids=[...document.querySelectorAll('[id]')].map(x=>x.id);
                  if(new Set(ids).size!==ids.length)throw Error('duplicate HTML IDs');
                  if(/undefined|NaN/.test(document.body.innerText))throw Error('invalid visible value');
                  el('dataset').value='cic2018';el('case').value='uniform';resetStages();
                  return {states:checked.length,summary_rows:el('summary').tBodies[0].rows.length,checked};
                })()''')
                if evaluate('Boolean(DATA.cumulative)'):
                    comparison=evaluate(r'''(()=>{
                      const checked=[];
                      for(const ds of ['cic2018','toniot']) for(const c of ['uniform','natural']) for(const m of ['xgb','tabpfn']){
                        el('cmp-dataset').value=ds;el('cmp-case').value=c;el('cmp-method').value=m;resetComparison();
                        const rr=cmpRecords(ds,c,m),br=records(ds,c,m);
                        for(const r of rr){
                          el('cmp-stage').value=r.stage;renderComparison();
                          const expected=[...r.classes].sort((a,b)=>b.support-a.support);
                          const actual=[...el('comparison-classes').tBodies[0].rows];
                          if(actual.length!==expected.length)throw Error('comparison class row count');
                          actual.forEach((row,i)=>{
                            const v=[...row.cells].map(x=>x.textContent),q=expected[i];
                            if(v[0]!==N[q.name]||v[1]!==num(q.support)||v[2]!==num(br[r.stage].fit_class_counts[q.name])||v[3]!==num(r.fit_class_counts[q.name])||v[9]!==pct(q.f1))throw Error('comparison metric mismatch');
                          });
                          if(el('comparison-stages').tBodies[0].rows.length!==rr.length)throw Error('comparison stages');
                          checked.push([ds,c,m,r.stage]);
                        }
                      }
                      if(el('comparison-summary').tBodies[0].rows.length!==8)throw Error('comparison summary');
                      if(/undefined|NaN/.test(document.body.innerText))throw Error('comparison invalid value');
                      el('cmp-dataset').value='toniot';el('cmp-case').value='natural';el('cmp-method').value='tabpfn';resetComparison();
                      return {states:checked.length,summary_rows:8,checked};
                    })()''')
                    result['comparison']=comparison
                    evaluate("el('comparison-classes').scrollIntoView();window.scrollBy(0,-80)")
                    screenshot('desktop-cumulative-classes.png')
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
