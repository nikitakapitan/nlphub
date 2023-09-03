
from datasets import load_metric
import transformers
import datasets


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="BERT baseline"):
        assert isinstance(pipeline, transformers.Pipeline)
        assert isinstance(dataset, datasets.Dataset)

        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        
        accuracy_score = load_metric("accuracy") # expects integers
        # prepare preds & labels
        preds, labels = [], []
        for example in self.dataset:
            pred = self.


    def compute_size(self):
        # We'll define this later
        pass

    def time_pipeline(self):
        # We'll define this later
        pass 

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics