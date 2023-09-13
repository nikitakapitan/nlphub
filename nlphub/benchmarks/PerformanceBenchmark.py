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
    def __init__(self, pipeline, metric_names):
        assert isinstance(pipeline, transformers.Pipeline)

        self.pipeline = pipeline
        self.dataset = None # task specific
        self.metrics = [load_metric(name) for name in metric_names]
                
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
        metrics = {
            **self.compute_size(),
            **self.compute_time(),
            **self.compute_performance(self.dataset, self.metrics),
        }

        return metrics
    

        