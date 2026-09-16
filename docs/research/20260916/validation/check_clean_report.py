import json,sys
from pathlib import Path
sys.path.insert(0,'/tmp/imbalcic_report_browser')
from playwright.sync_api import sync_playwright
repo=Path('/home/user/Desktop/imbalcic'); dest=repo/'docs/research/20260916/figures'
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1080},device_scale_factor=1)
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((repo/'lablog/html_report/clean_revalidation_0916.html').as_uri(),wait_until='networkidle')
 assert page.locator('#accepted-count').inner_text()=='36,156건'
 assert page.locator('#decided-count').inner_text()=='6'
 assert page.locator('#neutral-count').inner_text()=='36,150건'
 assert page.locator('#m-h').inner_text()=='5,720'
 assert page.locator('#m-d').inner_text()=='804,905'
 page.screenshot(path=str(dest/'clean_revalidation_desktop.png'),full_page=True)
 page.locator('#calibration').screenshot(path=str(dest/'calibration_explainer.png'))
 page.locator('#pre-select').select_option('0')
 assert page.locator('#decided-count').inner_text()=='9'
 page.locator('#post-select').select_option('6')
 assert page.locator('#decided-count').inner_text()=='0'
 page.locator('#heatmap').evaluate("e => e.closest('details').open = true")
 page.locator('#heatmap button[data-pre="0"][data-post="0"]').click()
 assert page.locator('#decided-count').inner_text()=='9'
 page.locator('#class-select').select_option('benign');assert page.locator('#m-d').inner_text()=='804,855'
 page.locator('#class-select').select_option('infiltration');assert page.locator('#m-h').inner_text()=='4,827'
 page.locator('#expert-select').select_option('expert2');page.locator('#class-select').select_option('benign');assert page.locator('#m-h').inner_text()=='3,898'
 page.locator('#split-select').select_option('cal');assert '2,816' in page.locator('#diagnostic-table').inner_text()
 with page.expect_download() as dl:page.locator('#download-chart').click()
 dl.value.save_as(str(dest/'exp47_calibration_diagnostic.svg'))
 page.locator('#split-select').select_option('eval')
 with page.expect_download() as dl:page.locator('#download-chart').click()
 dl.value.save_as(str(dest/'exp47_test_diagnostic.svg'))
 page.set_viewport_size({'width':390,'height':844});page.goto((repo/'lablog/html_report/clean_revalidation_0916.html').as_uri())
 assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'mobile document overflows'
 page.locator('#expert').screenshot(path=str(dest/'expert_mobile.png'))
 page.set_viewport_size({'width':1440,'height':1080})
 page.goto((repo/'lablog/html_report/post_tabpfn.html').as_uri()+'#clean_revalidation_0916')
 frame=page.frame_locator('#viewer');assert frame.locator('#accepted-count').inner_text()=='36,156건'
 assert page.locator('.nav-item[data-id="clean_revalidation_0916"]').count()==1
 page.screenshot(path=str(dest/'combined_report.png'))
 assert not errors, errors
 print(json.dumps({'passed':True,'errors':errors,'checks':['77-grid controls','expert/class matrix counts','cal/test diagnostic switch','SVG export','mobile overflow','combined embedded report']},ensure_ascii=False))
 browser.close()
