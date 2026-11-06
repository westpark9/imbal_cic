"""Count conflict flows with each side column behind the netFound vector, and read saved predictions; no model is fitted."""
import hashlib, json, sys, numpy as np, pandas as pd
from pathlib import Path
A=Path('tabpfn/results/v7/exp95/20261106_051907_all_class_0ecc759e'); R=Path('tabpfn/results/v7/exp94/20261104_104228_pcap_cic2017_9faba40e')
names=['benign','dos','portscan','ddos','bf','bot','web','inf','heartbleed']
a=np.load(A/'base/arrays.npz'); tail=np.load(A/'base/residual.npz')['tail']; kept=~tail
y7=a['y']; y9=np.load(A/'y9.npy'); split=a['split']; pk=a['packets']
Z=np.round(np.load(R/'encoders/netfound-v2-small/dataset.npz')['X'],3)+0.0
hb=lambda M: np.fromiter((int.from_bytes(hashlib.blake2b(v.tobytes(),digest_size=8).digest(),'little') for v in np.ascontiguousarray(M)),np.uint64,len(M))
zk=hb(Z); s=np.load(A/'side.npz'); sn=[str(v) for v in s['names']]; S=s['X']; print('side',sn,S.shape,len(y9),int(kept.sum()))
col=lambda c: S[:,sn.index(c)]
def stat(cols,y):
    # side.npz holds the rows in the same order as the full EXP94 table or the kept table
    d=pd.DataFrame({'k':zk[kept],'y':y[kept]})
    for i,c in enumerate(cols): d[f's{i}']=sideK[c]
    d['pk']=pk[kept]; d=d[d.y>=0]
    keys=['k']+[f's{i}' for i in range(len(cols))]
    nu=d.groupby(keys)['y'].transform('nunique').values
    g=d.groupby(keys+['y']).size().rename('n').reset_index(); t=g.groupby(keys).n.agg(['sum','max','size']); c=t[t['size']>1]
    con=d[nu>1]
    return dict(rows=len(d),conflict=int((nu>1).sum()),pct=round(100*(nu>1).mean(),3),groups=len(c),minority=int((c['sum']-c['max']).sum()),by_class={names[k]:int(v) for k,v in con.y.value_counts().items()},two_packet=int((con.pk<=2).sum()))
n=int(kept.sum())
sideK={c:(col(c) if len(S)==n else col(c)[kept]) for c in sn}
out={}
for lab,y in (('seven_class_labels',y7),('nine_class_labels',y9)):
    out[lab]={}
    for nm,cols in (('none',[]),('seq',['first_seq_zero']),('seq_ack_arm',['first_seq_zero','first_ack_zero']),('ports_arm',['pair_ports_60']),('all_three',['first_seq_zero','first_ack_zero','pair_ports_60'])):
        out[lab][nm]=stat(cols,y); print(lab,nm,out[lab][nm])
# saved predictions, nine classes, n=50
idx=np.flatnonzero(kept); yk=y9[idx]; sk=split[idx]; t=(sk=='test')&(yk>=0); yt=yk[t]
def conf(path,yy):
    C=np.zeros((9,9))
    for sd in range(43,48):
        p=np.load(path(sd)); assert len(p)==len(yy); np.add.at(C,(yy,p),1)
    return C/5
Ms={m:conf(lambda sd:A/f'evaluations/netfound/jobs/s{sd}_n50_{m}/predictions.npy',yt) for m in ('xgb','tabpfn')}
z=np.load('tabpfn/results/v7/exp94/20261103_162600_csv_cic2017_fewshot_c2c3b851/dataset.npz'); yc=z['y'].astype(int); sc=z['split']; tc=(sc=='test')&(yc>=0)
Cs={m:conf(lambda sd:A/f'evaluations/csv/jobs/s{sd}_n50_{m}/predictions.npy',yc[tc]) for m in ('xgb','tabpfn')}
for lab,D in (('netfound',Ms),('csv',Cs)):
    for m,C in D.items():
        print(lab,m,'benign row -> pred',{names[j]:round(C[0,j],1) for j in range(9) if C[0,j]>=.5})
        for c in (4,5,1,3):
            print('   pred',names[c],'total',round(C[:,c].sum(),1),{names[i]:round(C[i,c],1) for i in range(9) if C[i,c]>=.5})
out['confusion_n50_mean']={f'{l}_{m}':np.round(C,1).tolist() for l,D in (('netfound',Ms),('csv',Cs)) for m,C in D.items()}
json.dump(dict(names=names,**out),open(sys.argv[1],'w'),indent=1)
