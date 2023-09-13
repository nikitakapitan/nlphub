from nlphub import PerformanceBenchmark
import datasets
from nlphub.utils import rename_dataset_label_key

"""The MODELBenchmark:
1. measure time, memory and the performance on input dataset.

"""

class ClassificationBenchmark(PerformanceBenchmark):

    def __init__(self, pipeline, dataset, metric_names):
        super().__init__(pipeline, metric_names)

        # parse the label regex from data set (ex. 'label')
        self._label = list(dataset.features.keys())[1]
        self.features = dataset.features[self._label]
        

    def compuet_performance(self, dataset) -> dict:
        assert isinstance(dataset, datasets.Dataset), 'dataset is not of type datasets.Dataset'
        rename_dataset_label_key(dataset)
        assert 'text' in dataset and 'label' in dataset, "dataset doesn't contain 'text' or 'label' attributes"

        preds, labels = [], []

        for example in self.dataset:
            pred = self.pipeline(example['text'])

            pred_int =  self.features.str2int(pred[self._label])

            preds.append(pred_int)
            labels.append(example['label'])

        metrics = {}
        for metric in self.metrics:
            metrics[metric.name] = metric.compute(predictions=preds, references=labels)
        return metrics
