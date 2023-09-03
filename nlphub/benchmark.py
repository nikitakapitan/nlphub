from abc import ABC, abstractclassmethod
from datasets import load_metric
import transformers
import datasets
import numpy as np
import torch
from time import perf_counter
from pathlib import Path

class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, config):
        assert isinstance(pipeline, transformers.Pipeline)
        assert isinstance(dataset, datasets.Dataset)

        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = config['optim_type']

        for metric in config['metrics']:
            if metric == 'accuracy':
                self.metric = load_metric("accuracy") # expects integers
            else:
                raise ValueError(f'Metric {metric} is not yet supported')
                

    def compute_performance(self) -> dict:
        # prepare preds & labels
        preds, labels = [], []
        for example in self.dataset:
            pred = self.pipeline(example['text'])

            preds.append(intents.str2int(pred['label']))
            labels.append(example['intent'])

        score = self.metric.compute(predictions=preds, references=labels)
        return {self.metric : score}

    def compute_size(self) -> dict:
        state_dict = self.pipeline.model.state_dict()
        tmp = Path("model.pt")
        torch.save(state_dict, tmp)
        size_mb = Path(tmp).stat().st_size / (1024 * 1024)
        tmp.unlink()
        print(f"Model size (MB) - {size_mb:.2f}")
        return {'size_mb' : size_mb}


    def time_pipeline(self) -> dict:
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
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics