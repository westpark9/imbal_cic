"""Browser validation of the concise two-case report and its embedded copy."""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/tmp/imbalcic_report_browser')
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[4]
evidence = root / 'docs/research/20260916'
data = json.loads((evidence / 'clean_revalidation_visual_data.json').read_text())
figures = evidence / 'figures'
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1280, 'height': 1000}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto((root / 'lablog/html_report/clean_revalidation_0916.html').as_uri(), wait_until='networkidle')
    assert page.locator('h1').inner_text() == '동일벡터상반라벨 제거 후 모델 병목 지점 관찰'
    assert page.locator('[role=tab]').all_text_contents() == ['CIC2018', 'ToN']
    for key, filename in [('cic', 'brief_cic2018.png'), ('ton', 'brief_ton.png')]:
        page.locator('#tab-' + key).click()
        case = page.locator('#case-' + key)
        assert case.is_visible()
        assert case.locator('section').count() == 3
        assert case.locator('h3').all_text_contents() == ['1정제 후 클래스별 샘플 수', '2Expert 구성', '3호출·승인 결과와 병목']
        d, b = data[key], data['brief'][key]
        for i, name in enumerate(d['meta']['class_names']):
            expected = [name] + [f"{next(r['after'] for r in d['class_counts'] if r['split'] == split and r['class'] == name):,}" for split in ['train', 'test']]
            assert page.locator(f'#{key}-data-table tbody tr').nth(i).locator('td').all_text_contents() == expected
        assert page.locator(f'#{key}-expert-table tbody tr').count() == d['meta']['n_experts']
        assert page.locator(f'#{key}-calls').inner_text() == f"{d['native']['calls']:,}"
        assert page.locator(f'#{key}-accepted').inner_text() == f"{d['native']['accepted']:,}"
        for i, stage in enumerate(b['cal_funnel']):
            cells = page.locator(f'#{key}-funnel tbody tr').nth(i).locator('td').all_text_contents()
            assert cells[1:] == [f"{stage[k]:,}" for k in ['rows', 'helpful', 'harmful']]
        page.screenshot(path=str(figures / filename), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), key + ' mobile overflow'
        page.set_viewport_size({'width': 1280, 'height': 1000})
    assert '0건인 경우가 아닙니다' in page.locator('#case-ton').inner_text()
    page.locator('#tab-ton').focus()
    page.keyboard.press('ArrowLeft')
    assert page.locator('#case-cic').is_visible()
    assert page.locator('#tab-cic').get_attribute('aria-selected') == 'true'
    page.emulate_media(media='print')
    assert page.locator('#case-cic').is_visible() and page.locator('#case-ton').is_visible()
    page.emulate_media(media='screen')
    page.goto((root / 'lablog/html_report/post_tabpfn.html').as_uri() + '#clean_revalidation_0916', wait_until='networkidle')
    frame = page.frame_locator('#viewer')
    assert frame.locator('h1').inner_text() == '동일벡터상반라벨 제거 후 모델 병목 지점 관찰'
    frame.locator('#tab-ton').click()
    assert frame.locator('#ton-accepted').inner_text() == '271,715'
    assert frame.locator('#case-ton').is_visible()
    page.screenshot(path=str(figures / 'brief_combined.png'))
    assert not errors, errors
    result = {'passed': True, 'js_errors': errors,
              'checks': ['Exact requested title', 'Two datasets with identical 3 sections',
                         'All 17 class train/test counts', '8 CIC and 7 ToN experts',
                         'Native calibration stage counts', 'Actual test calls and approvals',
                         'Dataset tabs and keyboard', 'Mobile width', 'Print includes both cases',
                         'Combined report title and dataset switching']}
    (evidence / 'brief_report_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    browser.close()
