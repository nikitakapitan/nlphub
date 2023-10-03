from nlphub import Trainer
from nlphub import  DistillationTrainingArguments, DistillationTrainer
import logging

class FineTuner(Trainer):

    def __init__(self, config):
        super().__init__(config)

    def compute_metrics_func(self):
        raise NotImplemented
