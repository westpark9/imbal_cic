#!/usr/bin/env python3
"""Embed one Korean font family in the 0911/0918 reports and their index."""
import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'lablog/html_report/assets'
TARGETS = ['dataset_quality_0911.html', 'scorer_verifier_target_0918.html', 'post_tabpfn.html']


def apply_typography(page):
    css = (ASSETS / 'report_typography.css').read_text()
    page = re.sub(r'<link\b[^>]*fonts\.googleapis\.com[^>]*>\s*', '', page)
    page = re.sub(r'<style id="report-typography">.*?</style>\s*', '', page, flags=re.S)
    return page.replace('</head>', '<style id="report-typography">'+css+'</style>\n</head>', 1)


def main():
    from fontTools import subset
    from fontTools.ttLib import TTFont
    from lxml import html
    codepoints = set(range(32, 127))
    codepoints.update(map(ord, (ROOT / 'scripts/build_exp56_report_section.py').read_text()))
    for path in (ROOT / 'lablog/html_report').glob('*.html'):
        doc = html.fromstring(path.read_text())
        for node in doc.xpath('//script | //style'):
            node.drop_tree()
        codepoints.update(map(ord, doc.text_content()))
    faces = []
    for weight, name in [(400, 'Regular'), (700, 'Bold')]:
        font = TTFont(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{name}.ttc', fontNumber=1)
        options = subset.Options()
        options.layout_features = ['*']
        sub = subset.Subsetter(options=options)
        sub.populate(unicodes=sorted(codepoints))
        sub.subset(font)
        if weight == 400:
            font.save(ASSETS / 'report_notosans_kr_regular.otf')
        font.flavor = 'woff'
        path = ASSETS / f'report_notosans_kr_{name.lower()}.woff'
        font.save(path)
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        faces.append('@font-face{font-family:"Report Sans KR";font-style:normal;font-weight:'
                     + str(weight)+';font-display:swap;src:url(data:font/woff;base64,'+data+') format("woff")}')
    faces.append('html body,body *{font-family:"Report Sans KR","Noto Sans CJK KR","Malgun Gothic","Apple SD Gothic Neo",sans-serif!important}')
    (ASSETS / 'report_typography.css').write_text('\n'.join(faces)+'\n')
    for name in TARGETS:
        path = ROOT / 'lablog/html_report' / name
        path.write_text(apply_typography(path.read_text()))
    print('Embedded Noto Sans CJK KR regular/bold in', ', '.join(TARGETS))


if __name__ == '__main__':
    main()
