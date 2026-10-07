#!/usr/bin/env python3
"""Fresh single-seed bank when EXP57's missing residual state cannot be replayed."""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

from exp59_residual_membership_oracle import SOURCE, analyze, write


def run(root, dataset):
    import faulthandler
    faulthandler.enable(all_threads=True)
    frozen=SOURCE/'source/tabpfn/scripts'
    sys.path.insert(0,str(frozen));sys.path.insert(0,str(SOURCE/'source/scripts'))
    from nfv3_conflict_clean import install_clean_loader
    import nfv3_v3_common as core
    import torch
    from threadpoolctl import threadpool_limits
    from exp59_residual_membership_oracle import scores,confusion
    spec=importlib.util.spec_from_file_location('exp59_parent_suite',frozen/'exp57_expert_quality_suite.py')
    suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
    old=SOURCE/f'{dataset}_s43';out=root/f'{dataset}_s43';fresh=out/'fresh_bank'
    fresh.mkdir(parents=True,exist_ok=True)
    for directory in ['predictions','probabilities','contexts']:(fresh/directory).mkdir(exist_ok=True)
    job=json.loads((old/'job.json').read_text())
    job['args'].update(out_root=str(fresh),models_dir=str(fresh/'models'),resume_dir=str(fresh/'resume'))
    write(fresh/'job.json',job)

    class Fresh(suite.Experiment):
        def model(self,name,clf,corr,tune=None):
            self.progress('predicting_raw',model=name)
            v=self.v;C=v['n_classes'];y=v['y_eval'];base=self.out/'probabilities'
            if tune is None:tune=v['batched_proba'](clf,v['X_tune'],name+'/tune')
            test=v['batched_proba'](clf,v['X_eval'],name+'/test').astype(np.float32)
            np.save(base/(name+'_tune.npy'),np.asarray(tune,dtype=np.float32))
            np.save(base/(name+'_test.npy'),test)
            pred=test.argmax(1).astype(np.int16);np.save(self.out/'predictions'/(name+'_raw.npy'),pred)
            if name=='global':self.g=pred
            s=scores(confusion(y,pred,C));model=name+'_raw'
            self.summaries.append(dict(model=model,macro_f1=s['macro_f1'],accuracy=s['accuracy'],rows=len(y)))
            for c,cl in enumerate(v['class_names']):
                self.perclass.append(dict(model=model,**{'class':cl},**{k:s[k][c] for k in ['support','TP','FP','FN','f1']}))
                positive=y==c;fpr,recall,_=roc_curve(positive,test[:,c],drop_intermediate=False)
                self.probability.append(dict(model=model,**{'class':cl},AP=float(average_precision_score(positive,test[:,c])),
                    R_at_FPR_001=float(recall[fpr<=.001].max()),R_at_FPR_01=float(recall[fpr<=.01].max())))
            self.save_tables();self.progress('model_complete',model=name)
            return np.asarray(tune).argmax(1),pred

        def run_bank_diagnostics(self,v):
            self.progress('fresh_bank_start',reason='old residual state unavailable; replayed global differed')
            with np.load(old/'shared_context_ids.npz') as z:
                for key,var in [('global_context','g_idx'),('anchor','anchor_idx'),('expert_pool','exp_idx')]:
                    np.testing.assert_array_equal(z[key],v[var])
            with np.load(old/'evaluation_identity.npz') as z:
                np.testing.assert_array_equal(z['ids'],v['eval_idx'])
                np.testing.assert_array_equal(z['y'],v['y_eval'])
            # Preserve the actual fitted Global, not merely its seed.
            v['glob'].save_fit_state(fresh/'global_fit_state.tabpfn_fit')
            original_build=v['build_bank']

            def build_and_record(K):
                bank=original_build(K);mu=bank['mu'];sig=v['sig']
                self.progress('full_residual_test_assignment')
                v['embed_stage']['tag']='exp59/full_residual_test'
                z=v['phi'].transform(v['X_eval'])
                np.save(out/'test_z.npy',z)
                original_transform=v['phi'].transform
                v['phi'].transform=lambda X,*a,**kw:z if X is v['X_eval'] else original_transform(X,*a,**kw)
                p0=v['corr0'].correct(np.load(fresh/'probabilities/global_test.npy',mmap_mode='r'),0.,v['temp'])
                residual=v['balanced_ce'](p0,v['y_eval'],v['w_bal'])
                full=sig.full(z,p0,v['y_eval'],np.minimum(residual,v['r_max']))
                distances=v['sq_dist_to_centroids'](full,mu)
                regions=distances.argmin(1).astype(np.int16)
                obs=v['sq_dist_to_centroids'](sig.observable(z,p0),mu[:,:sig.obs_dim]).argmin(1)
                np.save(out/'residual_test_regions.npy',regions);np.save(out/'residual_test_distances.npy',distances)
                state=dict(full_centroids=mu,w_bal=v['w_bal'],train_regions=bank['assign'],train_labels=v['y_exp'],
                           train_ids=v['exp_idx'],train_residual_clip=np.asarray(v['r_max']),global_temperature=np.asarray(v['temp']))
                for name,(mean,std) in sig.stats.items():state[name+'_mean']=mean;state[name+'_std']=std
                np.savez(out/'residual_state.npz',**state)
                import joblib
                joblib.dump(v['phi'].pca,out/'feature_pca.joblib')
                audit=dict(evaluation_mode='fresh_single_seed_bank',same_global_context_anchor_expert_pool_test_ids=True,
                    old_predictions_reused=False,full_residual_vs_observable_changed_rows=int((regions!=obs).sum()),
                    test_rows=len(regions),train_only_scaling=True,residual_clip=float(v['r_max']),global_temperature=float(v['temp']))
                write(out/'reconstruction_audit.json',audit);write(out/'RECONSTRUCTION_COMPLETE.json',audit)
                v['manifest_df'].to_csv(out/'split_manifest.csv',index=False)
                v['pool_audit'].to_csv(out/'train_pool_partition.csv',index=False)
                return bank

            v['build_bank']=build_and_record
            result=super().run_bank_diagnostics(v)
            np.testing.assert_array_equal(np.load(fresh/'fixed_regions.npz')['centroids'],np.load(out/'residual_state.npz')['full_centroids'][:,:v['sig'].obs_dim])
            return result

    install_clean_loader(core,job['args']['clean_manifest'])
    torch.set_num_threads(16);torch.set_num_interop_threads(4)
    torch.manual_seed(43);np.random.seed(43)
    code=(old/'executed_prefix.py').read_text()
    code=code.replace('    return _exp57.run_bank_diagnostics(locals())',
        '    return _exp57.run_bank_diagnostics(dict(locals(), sq_dist_to_centroids=sq_dist_to_centroids, balanced_ce=balanced_ce))')
    (out/'executed_fresh_prefix.py').write_text(code)
    module=types.ModuleType('exp59_fresh_base');module.__file__=str(frozen/'nfv3_v3_exp31_c0alloc.py')
    module._exp57=Fresh(fresh,job);sys.modules[module.__name__]=module
    exec(compile(code,str(out/'executed_fresh_prefix.py'),'exec'),module.__dict__)
    with threadpool_limits(16):module.run_exp29(types.SimpleNamespace(**job['args']))
    analyze(fresh,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',required=True)
    a=p.parse_args()
    try:run(a.root.resolve(),a.dataset)
    except BaseException:
        import traceback
        write(a.root/f'{a.dataset}_s43'/'FRESH_ERROR.json',dict(traceback=traceback.format_exc()))
        raise
