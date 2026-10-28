"""Source-figure excerpts and paired class heatmaps for the 2026-10-22 review.

Requires PyMuPDF for vector PDF excerpts, pandas, numpy, matplotlib.
No report PDFs are retained. Crops preserve the original figure contents.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pymupdf as fitz

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/lablog.py').exists())
BASE = ROOT/'docs/lablog'
OUT = BASE/'figures/references_20261022'
EVID = BASE/'evidence/references/20261021'
RUN = ROOT/'tabpfn/results/v7/exp87/20261022_163739_full_learning_curve_five_seeds_c59213f5'
# Rectangles are PDF points in top-left-origin coordinates; captions are rewritten
# in Korean with original figure number, PDF page, and a clickable source link.
CROPS = [
 ('survey','Tabular Foundation Models: A Systematic Surve.pdf',6,(37,48,508,644),'Figure 3','https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7531718'),
 ('boost','boostpfn_2503.01256.pdf',2,(107,66,500,239),'Figure 1','https://arxiv.org/abs/2503.01256'),
 ('local','localpfn_neurips2024.pdf',5,(108,65,507,235),'Figure 3','https://papers.nips.cc/paper_files/paper/2024/file/c40daf14d7a6469e65116507c21faeb7-Paper-Conference.pdf'),
 ('dist','distpfn_2605.04363.pdf',4,(54,60,557,170),'Figure 4','https://arxiv.org/abs/2605.04363'),
 ('tabfm','tabfm_2609.37959.pdf',2,(62,78,535,372),'Figure 2','https://arxiv.org/abs/2609.37959'),
 ('pfn35','tabpfn35_2609.17895.pdf',10,(75,150,523,581),'Figure 8','https://arxiv.org/abs/2609.17895'),
 ('imbalance','2605.21742.pdf',4,(311,35,566,308),'Figure 4','https://arxiv.org/abs/2605.21742'),
 ('ensemble','2605.18696.pdf',5,(150,70,493,289),'Figure 1','https://arxiv.org/abs/2605.18696'),
 ('iot','2604.11394.pdf',5,(154,115,447,453),'Figure 1','https://arxiv.org/abs/2604.11394'),
 ('lotbonc','../../20261015/lotbonc.pdf',5,(53,72,300,218),'Figure 4','https://doi.org/10.1145/3779208.3785276'),
 ('streams','streams_2610.05352.pdf',2,(72,65,541,180),'Figure 1','https://arxiv.org/abs/2610.05352'),
 ('streams_cache','streams_2610.05352.pdf',8,(71,232,542,341),'Figure 7','https://arxiv.org/abs/2610.05352'),
 ('icd','5_In_Context_Data_Distillation.pdf',3,(108,86,505,205),'Figure 2','https://arxiv.org/abs/2402.06971'),
]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for key,name,page,box,figure,url in CROPS:
        src=(EVID/name).resolve()
        if key=='lotbonc': src=BASE/'evidence/20261015/lotbonc.pdf'
        doc=fitz.open(src); rect=fitz.Rect(box); out=fitz.open()
        target=out.new_page(width=rect.width,height=rect.height)
        target.show_pdf_page(target.rect,doc,page-1,clip=rect)
        path=OUT/(key+'.pdf');out.save(path,garbage=4,deflate=True)
        manifest.append(dict(asset=str(path.relative_to(ROOT)),source=str(src.relative_to(ROOT)),
                             source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                             pdf_page=page,figure=figure,crop_top_left_pdf_points=box,url=url))
    data=pd.read_csv(RUN/'per_class.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    # Keep all backgrounds light enough for both dark green and dark red deltas.
    cmap=LinearSegmentedColormap.from_list('light_blue',['#ffffff','#eaf1f8','#bfd4eb'])
    cells=[]
    for dataset,label in [('cic2018','CIC2018'),('toniot','ToN-IoT')]:
        d=data[data.dataset==dataset]
        assert d.groupby(['n','method','class_name']).seed.nunique().eq(5).all()
        supports=d.groupby('class_name').support.first().sort_values(ascending=False)
        assert d.groupby('class_name').support.nunique().eq(1).all()
        ns=sorted(d.n.unique()); classes=list(supports.index)
        means=d.groupby(['class_name','n','method']).f1.mean()
        pfn=np.array([[100*means.loc[c,n,'pfn'] for n in ns] for c in classes])
        xgb=np.array([[100*means.loc[c,n,'xgb'] for n in ns] for c in classes])
        fig,ax=plt.subplots(figsize=(11.8,.61*len(classes)+1.9))
        im=ax.imshow(pfn,cmap=cmap,vmin=0,vmax=100,aspect='auto')
        for i,c in enumerate(classes):
            for j,n in enumerate(ns):
                delta=float(pfn[i,j]-xgb[i,j]);display=round(delta,1)
                color='#006b38' if display>0 else '#b11827' if display<0 else '#5b5b5b'
                ax.text(j,i-.14,f'{pfn[i,j]:.1f}',ha='center',va='center',fontsize=10.5,color='#18212e',fontweight='bold')
                ax.text(j,i+.18,f'({display:+.1f})' if display else '(0.0)',ha='center',va='center',fontsize=9.8,color=color)
                cells.append(dict(dataset=dataset,n=int(n),class_name=c,support=int(supports[c]),
                                  pfn_f1_pct=float(pfn[i,j]),xgb_f1_pct=float(xgb[i,j]),delta_pp=delta,seeds=5))
        pretty={'brute_force':'Brute force','web_attacks':'Web attacks','ddos':'DDoS','dos':'DoS','xss':'XSS','mitm':'MITM'}
        ax.set_yticks(range(len(classes)),[pretty.get(c,c.capitalize())+f'  [{supports[c]:,}]' for c in classes])
        ax.set_xticks(range(len(ns)),[f'{n:,}' for n in ns]);ax.set_xlabel('Labeled examples per attack class (n)')
        ax.set_title(label+' | TabPFN F1 (%) and (TabPFN - XGB, percentage points)',pad=15,fontsize=12)
        ax.set_xticks(np.arange(-.5,len(ns),1),minor=True);ax.set_yticks(np.arange(-.5,len(classes),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='both',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        cb=fig.colorbar(im,ax=ax,pad=.02,fraction=.023);cb.set_label('TabPFN F1 (%)')
        fig.text(.5,.025,'Mean over seeds 43-47 | Rows: test support descending [count] | Green: PFN higher; red: XGB higher',ha='center',fontsize=9.5)
        fig.subplots_adjust(left=.21,right=.95,bottom=.17,top=.88)
        fig.savefig(OUT/('class_'+dataset+'.pdf'),bbox_inches='tight');plt.close(fig)
    (EVID/'report_figure_sources_20261022.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (EVID/'report_class_values_20261022.json').write_text(json.dumps(cells,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'source_crops':len(manifest),'heatmap_cells':len(cells),'run':str(RUN.relative_to(ROOT))}))

if __name__=='__main__':main()
