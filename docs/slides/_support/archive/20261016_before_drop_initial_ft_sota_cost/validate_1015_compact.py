#!/usr/bin/env python3
"""Check the seven-slide artifact against measured result sources."""
from pathlib import Path
import json,re,hashlib,subprocess
from pptx import Presentation
from pptx.enum.text import PP_ALIGN

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
qa=ROOT/'docs/slides/kor/1015_labmeeting_qa'
m=json.loads((qa/'manifest.json').read_text())
result=json.loads((ROOT/m['new_source']).read_text())
out=ROOT/m['output'];pdf=out.with_suffix('.pdf');prs=Presentation(out)
assert len(prs.slides)==7 and m['evaluation']=='final_full_test'
assert m['current_results_include_initial_ids_finetuning']
jobs={(j['dataset'],j['method']):j for j in result['jobs']}
for source in m.get('additional_sources',[]):
    extra=json.loads((ROOT/source).read_text())
    jobs.update({(j['dataset'],j['method']):j for j in extra['jobs']})
assert m['cost_metric_slides_6_7']=='update_seconds' and m['distpfn_initial_ft'] is False
assert all(r['metric'] in ('f1','update') for r in m['chart_data'])
assert {r['method'] for r in m['chart_data'] if r['slide']==6}=={'xgb','pfn_pretrained','pfn_ft_accumulate'}
assert 'distpfn_ft' not in {r['method'] for r in m['chart_data']}
assert 'distpfn_pretrained' in {r['method'] for r in m['chart_data'] if r['slide']==7}
for ds in ['cic2018','toniot']:
    j=jobs[ds,'distpfn_pretrained'];base=jobs[ds,'pfn_pretrained']
    assert j['initial_finetuning_seconds']==0 and j['update_seconds']==base['update_seconds']
    assert j['restored_base_prediction_mismatches']==0
for source,digest in m['source_sha256'].items():
    assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
for row in m['chart_data']:
    j=jobs[row['dataset'],row['method']]
    v={'f1':100*j['metrics']['macro_f1'],'update':j['update_seconds'],
       'total':j['total_preparation_seconds']+j['serving_seconds']}[row['metric']]
    assert abs(row['value']-v)<1e-10
for dataset in ['cic2018','toniot']:
    run=(ROOT/m['new_source']).parent;meta=json.loads((run/'data'/dataset/'manifest.json').read_text())
    source=Path(meta['source'])
    import numpy as np
    y=np.load(source/'train_y.npy');names=meta['class_names']
    plans=json.loads((run/'data'/dataset/'plans.json').read_text())
    for row in [r for r in m['scenario_data'] if r['dataset']==dataset]:
        plan=plans[row['stage']-1];assert row['cumulative']==len(plan['released'])
        actual={names[c].replace('_',' ').lower():int((y[plan['incoming']]==c).sum()) for c in np.unique(y[plan['incoming']])}
        assert actual=={name.lower():n for name,n in zip(row['classes'],row['new_counts'])}
issues=[];overlaps=[]
for i,slide in enumerate(prs.slides,1):
    texts=[]
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text:
            assert not re.search(r'합니다|입니다|습니다|—',shape.text),(i,shape.text)
            assert shape.left>=0 and shape.top>=0 and shape.left+shape.width<=prs.slide_width+10000
            assert shape.top+shape.height<=prs.slide_height+10000
            texts.append(shape)
        if shape.has_table:
            for row in list(shape.table.rows)[1:]:
                for cell in list(row.cells)[2:]:
                    assert all(p.alignment==PP_ALIGN.RIGHT for p in cell.text_frame.paragraphs)
    for a,shape in enumerate(texts):
        for other in texts[a+1:]:
            w=min(shape.left+shape.width,other.left+other.width)-max(shape.left,other.left)
            h=min(shape.top+shape.height,other.top+other.height)-max(shape.top,other.top)
            if w>10000 and h>10000:overlaps.append((i,shape.text,other.text))
assert not overlaps,overlaps
assert hashlib.sha256((ROOT/m['source_deck']).read_bytes()).hexdigest()==m['source_deck_sha256']
pages=int(re.search(r'Pages:\s+(\d+)',subprocess.check_output(['pdfinfo',str(pdf)],text=True)).group(1));assert pages==7
text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
assert '1014' not in text and '미적용 사전 비교' not in text
assert '최종' in text and '제안 모델' in text
assert '준비 + 전체 추론' not in text and '비용은 로그눈금' not in text
(qa/'compact/rendered_text.txt').write_text(text)
validation=dict(revision=m['revision'],slides=7,pdf_pages=pages,plot_values_verified=len(m['chart_data']),
    scenario_stages_verified=len(m['scenario_data']),numeric_columns_right_aligned=True,
    initial_ids_finetuning_results=True,evaluation='final_full_test',historical_1014_removed_from_main=True,
    same_label_budget_including_validation=True,unintended_text_overlap_count=len(overlaps),source_deck_unchanged=True,
    pptx_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
    consistent_update_cost_slides_6_7=True,distpfn_without_finetuning=True,visual_review='pending')
for dest in [qa/'validation.json',qa/'compact/validation.json']:dest.write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(validation,ensure_ascii=False,indent=2))
