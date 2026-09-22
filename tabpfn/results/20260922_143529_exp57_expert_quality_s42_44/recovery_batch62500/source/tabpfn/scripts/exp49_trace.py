"""EXP49 observations reuse EXP47 hooks with explicit BoT-IoT vocabulary."""
from exp47_trace import Trace as CICTrace, instrument


class Trace(CICTrace):
    def record(self, stage, v):
        super().record(stage, v)
        if stage == 'inputs':
            assert v['args'].target_dataset == 'bot_iot'
            assert 'benign' in self.meta['class_names']
            self.meta['protected_classes'] = ['ddos', 'dos']
            self.meta['dataset'] = 'bot_iot'
            self.meta['experiment'] = 'EXP49'
