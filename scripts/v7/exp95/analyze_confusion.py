import json, sys, numpy as np
from pathlib import Path
A=Path('tabpfn/results/v7/exp95/20261106_051907_all_class_0ecc759e'); P=Path('tabpfn/results/v7/exp95/20261106_042628_drop_paired_418e36f3')
names=['benign','dos','portscan','ddos','bf','bot','web','inf','heartbleed']
print(sorted(p.name for p in (A/'base').iterdir())[:8])
a=np.load(A/'base/arrays.npz'); tail=np.load(A/'base/residual.npz')['tail']; idx=np.flatnonzero(~tail)
y=np.load(A/'y9.npy')[idx]; split=a['split'][idx]
t9=(split=='test')&(y>=0); y9=y[t9]; in7=y9<7
print('test9',t9.sum(),'test7',in7.sum(),{n:int((y9==i).sum()) for i,n in enumerate(names)})
out=[]
for s in range(43,48):
    p9=np.load(A/f'evaluations/netfound/jobs/s{s}_n50_tabpfn/predictions.npy')
    p7=np.load(P/f'evaluations/netfound-exp94-pca/jobs/with_benign_s{s}_n50_tabpfn/predictions.npy')
    assert len(p9)==len(y9) and len(p7)==in7.sum()
    # rows predicted as the two new classes, by true class
    new=p9>=7
    by={n:int((y9[new]==i).sum()) for i,n in enumerate(names)}
    pinf=int((p9==7).sum()); phb=int((p9==8).sum())
    # bf false positives in the 7-class and 9-class models on the same 7-class test rows
    yy=y9[in7]; q9=p9[in7]
    fp7=(p7==4)&(yy!=4); fp9=(q9==4)&(yy!=4)
    moved=int((q9[fp7]>=7).sum())
    tp7=int(((p7==4)&(yy==4)).sum()); tp9=int(((q9==4)&(yy==4)).sum())
    # benign alarms
    ben=yy==0
    out.append(dict(seed=s,pred_inf=pinf,pred_heartbleed=phb,new_class_true=by,
        benign_alarms_9=int((q9[ben]!=0).sum()),benign_alarms_7=int((p7[ben]!=0).sum()),
        benign_to_new=int((q9[ben]>=7).sum()),
        bf_fp_7=int(fp7.sum()),bf_fp_9=int(fp9.sum()),bf_fp7_moved_to_new=moved,bf_tp_7=tp7,bf_tp_9=tp9,
        bf_true_to_new=int((q9[yy==4]>=7).sum())))
for r in out: print(r)
m=lambda k: float(np.mean([r[k] for r in out]))
print({k:round(m(k),1) for k in ('pred_inf','pred_heartbleed','benign_alarms_7','benign_alarms_9','benign_to_new','bf_fp_7','bf_fp_9','bf_fp7_moved_to_new','bf_tp_7','bf_tp_9','bf_true_to_new')})
tot={n:round(float(np.mean([r['new_class_true'][n] for r in out])),1) for n in names}; print(tot, round(sum(tot.values()),1))
json.dump(dict(runs=[str(A),str(P)],note='Rows the nine-class TabPFN n=50 model predicts as inf or heartbleed, and bf false positives on the same test rows of the seven-class and nine-class models.',seeds=out),open(sys.argv[1],'w'),indent=1)
