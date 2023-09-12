from abc import ABC, abstractclassmethod
from datasets import load_metric
import transformers
import numpy as np
import torch
from time import perf_counter
from pathlib import Path

""" This is TASK-Agnostic Base class.
"""

class PerformanceBenchmark:
    def __init__(self, pipeline, config):
        assert isinstance(pipeline, transformers.Pipeline)

        self.pipeline = pipeline
        self.optim_type = config['optim_type']
        self.dataset = None # be initialized with child classes


        for metric in config['metrics']:
            if metric == 'accuracy':
                self.metric = load_metric("accuracy") # expects integers
            else:
                raise ValueError(f'Metric {metric} is not yet supported')
                
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
        return {'size_mb' : size_mb}


    def compute_time(self) -> dict:
        latencies = []
        for _ in range(100):
            start = perf_counter()
            _ = self.pipeline(self.dataset[0]['text'])
            latencies.append(perf_counter() - start)
        time_avg_ms = 1_000 * np.mean(latencies)
        time_std_ms = 1_000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {'time_avg_ms' : time_avg_ms, 'time_std_ms' : time_std_ms}

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = {
            **self.compute_size(),
            **self.compute_time(),
            **self.compute_performance(self.dataset),
        }

        return metrics
    

# prepare preds & labels
        