####################################### COLAB #####################################
%load_ext autoreload
%autoreload 2

from huggingface_hub import login
login("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")

####################################### PYTHON #####################################

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
    


VRAM_Usage = MODEL_SIZE * PRECISION * BATCH_SIZE * SEQ_LEN

# Example LLaMa 2 7B
MODEL_SIZE = 7_000_000
PRECISION = 2           # float16
BATCH_SIZE = 1          # single example
SEQ_LEN = 222_477       # number of tokens NarrativeQA[train][0]

# 7kk * 2 * 222k = 3_108_000_000_000 bytes