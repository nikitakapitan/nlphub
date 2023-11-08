Launch report.py on google colab.

*************Set up*************

Turn ON GPU
!git clone https://github.com/nikitakapitan/nlphub.git
!mv nlphub/report.yaml .
!mkdir logs
!pip install datasets transformers evaluate accelerate 

adapt report.yaml
adapt nlphub/report.py (if needed)

!python nlphub/report.py --config report.yaml


************* Classification Fine-tunning*************
!pip install accelerate transformers datasets evaluate

dataset = load_dataset('emotion')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

tokenize = lambda example : tokenizer(example['text'], padding='max_length', truncation=True)
tokenized_dataset = dataset.map(tokenize, batched=True)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_LABELS)

training_args = TrainingArguments(
    output_dir='distilbert-base-uncased-finetuned-clinc-oos',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics
)
trainer.train()

************* Evaluation *************
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







# Results:
As of 27Sep23 report info will be stored isnide the logs.