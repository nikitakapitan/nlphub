from abc import ABC, abstractclassmethod
import evaluate
import transformers
import numpy as np
import torch
from time import perf_counter
from pathlib import Path
import logging

""" This is TASK-Agnostic Base class.
"""

class PerformanceBenchmark(ABC):
    def __init__(self, pipeline, metric_cfgs):
        """
        pipeline = transformers.pipeline('task', 'model')
        metrics = [ {'name': 'accuracy', 'args': {}},
                    {'name': 'f1',       'args': {'average': 'weighted'}}]
        """
        assert isinstance(pipeline, transformers.Pipeline)
        self.pipeline = pipeline

        self._prepare_metrics(metric_cfgs)

        
    def _prepare_metrics(self, metric_cfgs):
        self.metric_funcs = {}
        for metric_cfg in metric_cfgs:
            metric_name = metric_cfg['name']
            metric_args = metric_cfg.get('args', {})
            self.metric_funcs[metric_name] = {
                'func': evaluate.load(metric_name),
                'args': metric_args         }

    @abstractclassmethod
    def compute_performance(self, dataset) -> dict:
        """Abstract method.

        Example:
        preds, labels = [], []
        for example in self.dataset:
            preds.append(self.pipeline(example['input']))
            labels.append(example['label'])

        score = self.metric.compute(predictions=preds, references=labels)
        return {self.metric.name : score}
        """
        pass

    def compute_size(self) -> dict:
        state_dict = self.pipeline.model.state_dict()
        tmp = Path("model.pt")
        torch.save(state_dict, tmp)
        size_mb = Path(tmp).stat().st_size / (1024 * 1024)
        tmp.unlink()
        print(f"Model size (MB) - {size_mb:.2f}")
        logging.info(f"Model size (MB) - {size_mb:.2f}")
        return {'size_mb' : size_mb}


    def compute_time(self) -> dict:
        # warm-up
        for _ in range(10):
            _ = self.pipeline(self.dataset[0]['text'])

        latencies = []
        for _ in range(100):
            start = perf_counter()
            _ = self.pipeline(self.dataset[0]['text'])
            latencies.append(perf_counter() - start)
        time_avg_ms = 1_000 * np.mean(latencies)
        time_std_ms = 1_000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        logging.info(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {'time_avg_ms' : time_avg_ms, 'time_std_ms' : time_std_ms}

    def run_benchmark(self):
        metrics = {
            **self.compute_size(),
            **self.compute_time(),
            **self.compute_performance(),
        }

        return metrics
    

        