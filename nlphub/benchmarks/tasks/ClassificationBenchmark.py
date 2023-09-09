from nlphub import PerformanceBenchmark
import datasets

"""The MODELBenchmark:
1. measure time, memory and the performance on input dataset.

New class should:
1. cover all tasks (ex. classification, ner etc)
"""

class ClassificationBenchmark(PerformanceBenchmark):

    def __init__(self, pipeline, dataset, config):
        super().__init__(pipeline, dataset, config)

        # parse the label regex from data set (ex. 'label')
        self._label = list(dataset.features.keys())[1]
        self.features = dataset.features[self._label]
        

    def compuet_performance(self, dataset) -> dict:
        assert isinstance(dataset, datasets.Dataset)

        preds, labels = [], []

        for example in self.dataset:
            pred = self.pipeline(example['text'])

            pred_int =  self.features.str2int(pred[self._label])

            preds.append(pred_int)
            labels.append(example['text'])

        score = self.metric.compute(predictions=preds, references=labels)
        return {self.metric : score}
