#!/usr/bin/env python3
"""Add a third BoT-IoT case (EXP49) to the existing CIC2018/ToN clean_revalidation_0916.html
report, reusing collect()/brief_case()/read_json()/read_csv() from
build_clean_revalidation_report.py unmodified. Does not touch that script or its CIC/ToN output;
only extends the already-rendered clean_revalidation_0916.html + its visual_data.json in place."""
import json
from pathlib import Path
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from build_clean_revalidation_report import collect, brief_case, read_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / 'lablog/html_report/clean_revalidation_0916.html'
VISUAL_DATA = HERE / 'clean_revalidation_visual_data.json'


def main():
    bot = collect(read_json(HERE / 'exp49_launch.json'))
    assert bot['state'] == 'complete', f"EXP49 not complete: {bot['state']}"
    bot_brief = brief_case(bot)

    data = json.loads(VISUAL_DATA.read_text())
    assert 'bot' not in data, 'bot case already present; refusing to overwrite'
    data['bot'] = bot
    data['brief']['bot'] = bot_brief
    data['asof'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')

    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', r'<\/')
    text = OUTPUT.read_text()
    match = re.search(r'(<script type="application/json" id="report-data">)(.*?)(</script>)', text, re.S)
    assert match, 'report-data script tag not found'
    text = text[:match.start(2)] + encoded + text[match.end(2):]

    # Third tab button, after the existing ToN button.
    assert text.count('<button id="tab-ton"') == 1
    text = re.sub(
        r'(<button id="tab-ton"[^>]*>ToN</button>)',
        r'\1\n<button id="tab-bot" role="tab" aria-controls="case-bot" aria-selected="false" '
        r'tabindex="-1" data-case="bot">BoT-IoT</button>',
        text)

    # renderCase() call: add the third case.
    text = text.replace(
        "document.getElementById('cases').innerHTML=renderCase('cic','CIC2018')+renderCase('ton','ToN');",
        "document.getElementById('cases').innerHTML=renderCase('cic','CIC2018')+renderCase('ton','ToN')+renderCase('bot','BoT-IoT');")

    # tailNote: was a cic/else-ton ternary; make it a per-case lookup including bot.
    old_tail = ("const tailNote=key==='cic'?'Web_attacks는 train 450행, test 127행이 남았습니다.'"
                ":'Mitm은 train 3,285행·test 1,161행, ransomware는 train 2,151행·test 709행이 남았습니다.';")
    assert old_tail in text
    new_tail = ("const tailNote={cic:'Web_attacks는 train 450행, test 127행이 남았습니다.',"
                "ton:'Mitm은 train 3,285행·test 1,161행, ransomware는 train 2,151행·test 709행이 남았습니다.',"
                "bot:'Theft는 train 388행, test 81행뿐입니다. Tail 클래스보다 ddos(train 4,181,543행)·dos(train 4,665,951행)가 "
                "F1 0에 가깝다는 점이 이 데이터셋의 실제 병목입니다.'}[key];")
    text = text.replace(old_tail, new_tail)

    # valid-key guard, arrow-key cycling, hash routing: extend from a 2-way ternary to the 3 known cases.
    text = text.replace("if(!['cic','ton'].includes(key))key='cic';", "if(!['cic','ton','bot'].includes(key))key='cic';")
    old_keydown = ("b.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){"
                   "e.preventDefault();const key=e.key==='Home'?'cic':e.key==='End'?'ton':"
                   "b.dataset.case==='cic'?'ton':'cic';selectCase(key);document.getElementById('tab-'+key).focus();}});")
    assert old_keydown in text
    new_keydown = ("b.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){"
                   "e.preventDefault();const order=['cic','ton','bot'];"
                   "const key=e.key==='Home'?'cic':e.key==='End'?'bot':"
                   "order[(order.indexOf(b.dataset.case)+(e.key==='ArrowLeft'?order.length-1:1))%order.length];"
                   "selectCase(key);document.getElementById('tab-'+key).focus();}});")
    text = text.replace(old_keydown, new_keydown)

    text = text.replace(
        "if(updateHash)history.replaceState(null,'','#'+(key==='cic'?'cic2018':'ton'));",
        "if(updateHash)history.replaceState(null,'','#'+(key==='cic'?'cic2018':key));")
    text = text.replace(
        "window.addEventListener('hashchange',()=>selectCase(location.hash==='#ton'?'ton':'cic',false));\n"
        "selectCase(location.hash==='#ton'?'ton':'cic',false);",
        "window.addEventListener('hashchange',()=>selectCase(['ton','bot'].includes(location.hash.slice(1))?location.hash.slice(1):'cic',false));\n"
        "selectCase(['ton','bot'].includes(location.hash.slice(1))?location.hash.slice(1):'cic',false);")

    # asof line and header badge counts.
    text = text.replace(
        "document.getElementById('asof').textContent='실험 완료: CIC2018 09-15 · ToN 09-16. 보고서 갱신: '+D.asof.slice(0,16).replace('T',' ')+' KST. 저장 결과를 재구성했으며 새 학습은 실행하지 않았습니다.';",
        "document.getElementById('asof').textContent='실험 완료: CIC2018 09-15 · ToN 09-16 · BoT-IoT 09-17. 보고서 갱신: '+D.asof.slice(0,16).replace('T',' ')+' KST. 저장 결과를 재구성했으며 새 학습은 실행하지 않았습니다.';")

    OUTPUT.write_text(text)
    VISUAL_DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'Added bot case. macro_global={bot["macro_global"]:.6f} native_macro={bot["native"]["macro_f1"]:.6f} '
          f'calls={bot["native"]["calls"]} accepted={bot["native"]["accepted"]}')


if __name__ == '__main__':
    main()
