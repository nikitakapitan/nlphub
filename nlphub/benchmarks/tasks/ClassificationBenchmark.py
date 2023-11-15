from nlphub import PerformanceBenchmark


import logging

"""The MODELBenchmark:
1. measure time, memory and the performance on input dataset.

"""

class ClassificationBenchmark(PerformanceBenchmark):

    def __init__(self, pipeline, dataset, metric_config):
        """
        pipeline    : transformers.pipeline
        dataset     : datasets.Dataset
        metric_cfgs : list[dict]
                    [{'accuracy'   : {}},
                     {'f1'         : {'average': 'weighted'}},
                     {'glue'       : ['cola', 'mrpc', 'qqp']}]
        """
        super().__init__(pipeline, dataset, metric_config)
        

    def compute_performance(self) -> dict:
        preds, labels = [], []

        for example in self.dataset:
            pred = self.pipeline(example['text'])
            pred_int =  self.pipeline.model.config.label2id[pred[0]['label']]
            preds.append(pred_int)
            labels.append(example['label'])

        metrics = {}
        for metric_func, metric_args in self.metrics_functions.items():
            metrics.update(metric_func.compute(predictions=preds, references=labels, **metric_args))
    
        return metrics
