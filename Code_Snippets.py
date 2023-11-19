%load_ext autoreload
%autoreload 2

#### slice dataset:
from datasets import Dataset
        slice_dataset_dict = tokenized_datasets['test'][:10]
        slice_dataset = Dataset.from_dict(slice_dataset_dict)

#### manual accuracy:
from evaluation import load
accuracy_metric = load('accuracy')

test_pred = trainer.predict(tokenized_datasets['test'])
logits = test_pred.predictions
labels = test_pred.label_ids
predictions = np.argmax(logits, axis=-1)

accuracy_metric.compute(predictions=predictions, references=labels)


# parse yaml
with open(args, 'r') as f:
        config = yaml.safe_load(f)
    