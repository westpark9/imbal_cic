"""Frozen, pinned netFound-v2 inference with padding-invariant mean pooling."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from fetch_assets import SOURCE, sha256

TOKEN_KEYS = ['flow_duration','burst_tokens','directions','bytes','iats','counts','protocol']

class NetFoundEncoder:
    def __init__(self, root, checkpoint, device='cuda', dtype='float32'):
        os.environ.setdefault('USE_TF', '0')
        source = Path(root)/SOURCE['path']
        head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if head != SOURCE['commit']: raise ValueError('netFound source does not match pinned commit')
        sys.path.insert(0, str(source/'src'))
        from modules.netFoundConfigBase import netFoundConfig
        from modules.netFoundModels import netFoundLanguageModelling
        from modules.netFoundTokenizer import netFoundTokenizer
        from modules.netFoundDataCollator import SimpleDataCollator
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: use a driver-compatible PyTorch or explicitly choose --device cpu')
        self.device, self.dtype = torch.device(device), getattr(torch,dtype)
        config = netFoundConfig.from_pretrained(checkpoint, local_files_only=True)
        config.pretraining, config.use_flash_attn, config.compile = False, False, False
        if not config.strip_payload: raise ValueError('This adapter is validated for v2 header-only checkpoints')
        model, info = netFoundLanguageModelling.from_pretrained(checkpoint,config=config,local_files_only=True,output_loading_info=True)
        if any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
            raise RuntimeError(f'Checkpoint mismatch: {info}')
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.model.requires_grad_(False)
        self.tokenizer = netFoundTokenizer(config)
        self.tokenizer.pretraining = True
        self.collator = SimpleDataCollator(pad_token_id=0)
        self.metadata = dict(model='netFound-v2-small', hidden_size=config.hidden_size,
                             source_commit=head, checkpoint_sha256=sha256(Path(checkpoint)/'model.safetensors'),
                             config_sha256=sha256(Path(checkpoint)/'config.json'),
                             dtype=dtype, device=device, pooling='valid-token mean, includes burst CLS, excludes padding',
                             payload=False, trainable_parameters=0,
                             total_parameters=sum(p.numel() for p in model.parameters()))

    @torch.inference_mode()
    def encode(self, rows):
        data = {k:[copy.deepcopy(r[k]) for r in rows] for k in TOKEN_KEYS}
        data = self.tokenizer(data)
        examples = [{k:v[i] for k,v in data.items()} for i in range(len(rows))]
        batch = self.collator(examples)
        keys = ['input_ids','attention_mask','direction','iats','bytes','pkt_count','protocol','dataset_burst_sizes']
        args = {k:batch[k].to(self.device, dtype=self.dtype if k in ['direction','iats','bytes','pkt_count'] else torch.long) for k in keys}
        args['position_ids'] = torch.arange(args['input_ids'].shape[1],device=self.device).unsqueeze(0).expand(len(rows),-1)
        hidden = self.model.base_transformer(**args,return_dict=True).last_hidden_state.float()
        mask = args['attention_mask'].unsqueeze(-1).float()
        embeddings = (hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        result = embeddings.cpu().numpy()
        if not np.isfinite(result).all(): raise FloatingPointError('Non-finite encoder output')
        return result
