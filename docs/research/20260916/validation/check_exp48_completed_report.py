import sys
sys.path.insert(0,'/tmp/imbalcic_report_browser')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1200,'height':900});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('file:///tmp/exp48_completed_report_preview.html');assert '완료' in page.locator('#ton-status').inner_text();assert '실제 최종 정책' in page.locator('#ton-result').inner_text()
 page.locator('#ton-expert-select').evaluate("e => e.closest('details').open = true")
 page.locator('#ton-expert-select').select_option('expert2');page.locator('#ton-expert-class').select_option('benign');assert '140행' in page.locator('#ton-expert-matrix').inner_text()
 assert len(page.locator('#ton-class-table tr').all())==10
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
 assert not errors, errors;print('Completed ToN report: all 10 classes, actual/diagnostic values, expert matrix, mobile and JS passed.');b.close()
