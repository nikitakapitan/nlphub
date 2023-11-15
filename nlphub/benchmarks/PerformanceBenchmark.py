from abc import ABC, abstractclassmethod
import evaluate
import transformers
import numpy as np
import torch
from time import perf_counter
from pathlib import Path
import logging
import datasets
from nlphub.utils import rename_split_label_key

""" This is TASK-Agnostic Base class.
"""

class PerformanceBenchmark(ABC):
    def __init__(self, pipeline, dataset, metric_config):
        """
        pipeline = transformers.pipeline('task', 'model')
        metric_cfgs : list[dict]
                    [{'accuracy'   : {}},
                     {'f1'         : {'average': 'weighted'}},
                     {'glue'       : ['cola', 'mrpc', 'qqp']}]
        """
        self.dataset = dataset
        self.pipeline = pipeline
        self.sanityCheck()

        self._prepare_metrics(metric_config)

    def sanityCheck(self):
        assert isinstance(self.dataset, datasets.Dataset), \
        f'dataset is not of type datasets.Dataset but {type(self.dataset)}'
        self.dataset = rename_split_label_key(self.dataset)
        assert ('text' in self.dataset.features) and ('label' in self.dataset.features), \
        f"dataset doesn't contain 'text' and 'label' but {self.dataset.features.keys()}"

        assert isinstance(self.pipeline, transformers.Pipeline), \
        f'pipeline is not of type datasets.Dataset but {type(self.pipeline)}'
        
    def _prepare_metrics(self, metric_config):
        """output is a dict keys = functions, values = configs:
        {evaluate.metrics.accuracy : {},
         evaluate.metrics.f1_score : {average: weighted}
        """
        metrics_functions = {}
        for metric_dict in metric_config:
            for metric_name, metric_cfg in metric_dict.items():
                if metric_name == "glue":
                    for glue_task in metric_cfg:
                        metrics_functions[evaluate.load("glue", glue_task)] = {}
                else:
                    metrics_functions[evaluate.load(metric_name)] = metric_cfg
            
        self.metrics_functions = metrics_functions

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
        return {'PerformanceBenchmark.py: time_avg_ms' : time_avg_ms, 'time_std_ms' : time_std_ms}

    def run_benchmark(self):
        metrics = {
            **self.compute_size(),
            **self.compute_time(),
            **self.compute_performance(),
        }

        return metrics
    

        