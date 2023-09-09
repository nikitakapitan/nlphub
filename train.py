# python train.py --config config.yaml

# requirements:
# !pip install datasets transformers evaluate
import os
import yaml
import argparse
import logging
import time

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
import evaluate
from nlphub.benchmarks.mapping import task_to_benchmark, task_to_auto_model

# Initialize logging
logging.basicConfig(filename=f"logs/train_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log", level=logging.INFO)


def rename_dataset_label_key(dataset):
    for key, feature in dataset['train'].features.items():
        if isinstance(feature, datasets.ClassLabel):
            current_key = key
    if current_key != 'label':
        for split in dataset.keys():
            dataset[split] = dataset[split].rename_column(current_key, 'label')


def get_dataset_num_classes(features):
    "Extract num_classes from provided dataset"
    for key, feature in features.items():
        if hasattr(feature, 'num_classes'):
            return features[key].num_classes
    raise ValueError("Could not find a suitable label key with num_classes in the dataset features ")


def main(args):

    with open(args, 'r') as f:
        config = yaml.safe_load(f)

    # Log input configs
    logging.info("Input Configurations:")
    logging.info(yaml.dump(config))

    os.environ['TRANSFORMERS_TOKEN'] = config['HF_TOKEN']

    # Dynamic Class Mapping
    AutoModelClass = task_to_auto_model[config['TASK']]
    logging.info(f"Mapped to AutoModel Class: {AutoModelClass.__name__}")

    device = 'cuda'

    # Load Dataset
    try:
        dataset_config_name = config.get('DATASET_CONFIG_NAME') # can be None
        dataset = load_dataset(config['DATASET_NAME'], dataset_config_name)
        rename_dataset_label_key(dataset)
        logging.info(f"Dataset {config['DATASET_NAME']} loaded.")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        exit(1)

    # Tokenization
    try:
        tokenizer = AutoTokenizer.from_pretrained(config['BASE_MODEL_NAME'])
        logging.info(f"Tokenizer for {config['BASE_MODEL_NAME']} initialized.")
    except Exception as e:
        logging.error(f"Error initializing tokenizer: {e}")
        exit(1)

    tokenize = lambda batch: tokenizer(batch['text'], truncation=True)
    dataset_encoded = dataset.map(tokenize, batched=True)

    # Model Initialization
    try:
        num_classes = get_dataset_num_classes(dataset['train'].features)
        model = AutoModelClass.from_pretrained(config['BASE_MODEL_NAME'], num_labels=num_classes)
        logging.info(f"Model {model.__class__.__name__} initialized with {num_classes} classes.")
        model.to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit(1)

    # Load metrics
    metrics = {}
    for metric_config in config['METRIC_NAMES']:
        metric_name = metric_config['name']
        metric_args = metric_config.get('args', {})
        try:
            metrics[metric_name] = {
                'func': evaluate.load(metric_name),
                'args': metric_args
                }
        except Exception as e:
            print(f"Error loading metric {metric_name}: {e}")
            exit(1)

    def compute_eval_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)

        results = {}
        for metric_name, metric_detail in metrics.items():
            metric_func = metric_detail['func']
            metric_args = metric_detail['args']
            results[metric_name] = metric_func.compute(predictions=preds, references=labels, **metric_args)
            
        return results


    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f'{config["BASE_MODEL_NAME"]}-finetuned-{config["DATASET_NAME"]}',
        num_train_epochs=config['NUM_EPOCHS'],
        learning_rate=config['LEARNING_RATE'],
        per_device_train_batch_size=config['BATCH_SIZE'],
        per_device_eval_batch_size=config['BATCH_SIZE'],
        weight_decay=0.01,
        evaluation_strategy='epoch',
        disable_tqdm=False,
        logging_dir='./logs',
        push_to_hub=True,
        log_level=config['LOG_LEVEL'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_encoded['train'],
        eval_dataset=dataset_encoded[config['EVAL_DATASET']],
        tokenizer=tokenizer,
        compute_metrics=compute_eval_metrics,
    )

    # Train and Evaluate
    try:
        trainer.train()
        trainer.evaluate()
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        exit(1)

    # Push to Hub
    try:
        trainer.push_to_hub()
        logging.info("Model pushed to Hugging Face Hub.")
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning models with Hugging Face Transformers')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
    args = parser.parse_args()
    main(args.config)