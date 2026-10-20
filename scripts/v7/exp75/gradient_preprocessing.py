"""Finite-gradient standardization for the isolated ICD process.

The upstream nanstd squares NaNs before masking and differentiates sqrt(0).
Use a masked vector norm (finite zero subgradient) instead. Ordinary TabPFN
workers and the installed package files are not changed.
"""
import torch
from tabpfn.preprocessing.steps.differentiable_z_norm_step import DifferentiableZNormStep
from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler


def stable_statistics(x, axis=0):
    mask = torch.isnan(x)
    clean = torch.where(mask, torch.zeros_like(x), x)
    count = (~mask).sum(dim=axis)
    mean = clean.sum(dim=axis)/count.clamp_min(1)
    centered = torch.where(mask, torch.zeros_like(clean), clean-mean.unsqueeze(axis))
    std = torch.linalg.vector_norm(centered, dim=axis)/(count-1).clamp_min(1).to(x.dtype).sqrt()
    return mean, std


class StableDifferentiableZNormStep(DifferentiableZNormStep):
    def _fit(self, X, feature_schema):
        mean, std = stable_statistics(X)
        self.means = mean.unsqueeze(0)
        self.stds = torch.where(std == 0, torch.ones_like(std), std).unsqueeze(0)
        return feature_schema


class StableStandardScaler(TorchStandardScaler):
    def fit(self, x):
        mean, std = stable_statistics(x)
        std = torch.where(std == 0, torch.ones_like(std), std)
        if x.shape[0] == 1: std = torch.ones_like(std)
        return dict(mean=mean, std=std)


def enable_context_preprocessor():
    # The factory stores its imported class as a module attribute. Replace that
    # attribute only in this ICD worker; no library file/global service is edited.
    import tabpfn.preprocessing.pipeline_factory as factory
    factory.DifferentiableZNormStep = StableDifferentiableZNormStep
    # Outlier clipping creates NaNs internally before recomputing dispersion.
    # Its imported nanstd alias must also use masking before squaring.
    import tabpfn.preprocessing.torch.torch_soft_clip_outliers as clipping
    clipping.torch_nanstd = lambda x, axis=0: stable_statistics(x, axis)[1]


def stabilize_backbone_scaler(model):
    for network in model.models_:
        if not hasattr(network, 'standard_scaler'):
            raise TypeError('Expected the explicitly supported TabPFN-v3 scaler')
        network.standard_scaler = StableStandardScaler()
