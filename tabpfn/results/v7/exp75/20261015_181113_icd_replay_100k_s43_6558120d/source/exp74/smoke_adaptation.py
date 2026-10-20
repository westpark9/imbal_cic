#!/usr/bin/env python3
"""Real-data GPU smoke: head serialization and frozen-backbone context gradients."""
from run_adaptation import *


def main():
    import torch
    from tabpfn import TabPFNClassifier
    from distribution_heads import Head
    args = argparse.ArgumentParser(); args.add_argument('--out', type=Path, required=True)
    args = args.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    config = defaults('heads'); torch.set_num_threads(8)
    config.update(head_epochs=3, head_patience=2, context_budget=64,
                  icd_steps=2, icd_eval_every=1, icd_query_batch=16, icd_eval_batch=64)
    data, identity = stage_data(PREPARED, config, 'toniot', 'natural', 5)
    rng = np.random.default_rng(43); C = len(data['names'])
    # The head smoke uses a small real-input context to keep verification fast.
    take = np.concatenate([rng.permutation(np.flatnonzero(data['ytr'] == c))[:16] for c in range(C)])
    base = TabPFNClassifier(device='cuda:0', model_path=config['checkpoint'], n_estimators=2,
        auto_scale_n_estimators=False, random_state=43, fit_mode='fit_with_cache',
        keep_cache_on_device=False, ignore_pretraining_limits=True)
    base.fit(data['Xtr'][take], data['ytr'][take])
    def few(y): return np.concatenate([np.flatnonzero(y == c)[:8] for c in range(C)])
    tr = few(data['y_tune']); va = few(data['y_cal'])
    zfit = embed(base, data['X_tune'][tr], 128); zcal = embed(base, data['X_cal'][va], 128)
    path = out/'heads'; path.mkdir(exist_ok=True)
    fit_heads(zfit, data['y_tune'][tr], zcal[::2], data['y_cal'][va][::2],
              zcal[1::2], data['y_cal'][va][1::2], np.bincount(data['ytr']), config, path)
    predictor = HeadsPredictor(path, base)
    p1, aux = predictor.predict_with_aux(data['X_cal'][va[:16]])
    p2, _ = predictor.predict_with_aux(data['X_cal'][va[:16]])
    np.testing.assert_allclose(p1, p2, atol=1e-6)
    np.testing.assert_allclose(p1.sum(1), 1, atol=1e-6)
    assert set(aux) == set(HEAD_NAMES)
    predictor.close(); del predictor, base; gc.collect(); torch.cuda.empty_cache()
    data['X_cal'] = data['X_cal'][va]; data['y_cal'] = data['y_cal'][va]
    path = out/'icd'; path.mkdir(exist_ok=True)
    fit_context(data, config, path, lambda **x: print(x, flush=True))
    model = ContextPredictor(path, config, C)
    p, _ = model.predict_with_aux(data['X_cal'][:16])
    assert np.isfinite(p).all(); np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)
    model.close()
    write(out/'SMOKE_COMPLETE.json', dict(complete=True, data='ToN natural final stage, train/validation only',
        head_save_load=True, two_estimator_embeddings=True, prediction_repeatability=True,
        context_gradient_steps=2, frozen_backbone=True, no_test_access=True))


if __name__ == '__main__': main()
