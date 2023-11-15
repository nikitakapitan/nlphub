# Colab:
# Turn ON GPU

#%%capture
#!git clone https://github.com/nikitakapitan/nlphub.git
#!mv nlphub/train.yaml .
#!mkdir logs
#!pip install datasets transformers evaluate accelerate 
# from huggingface_hub import login
# login("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")

# >>> Customize train.yaml

# python nlphub/train.py --config train.yaml
 
import os
import yaml
import logging
import argparse
import time
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from nlphub import FineTuner

# Initialize logging
if not os.path.exists('/content/logs/'):
    os.makedirs('/content/logs/')
logging.basicConfig(filename=f"logs/train_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log", level=logging.INFO)

def main(args):

    with open(args, 'r') as f:
        config = yaml.safe_load(f)
    finetuner = FineTuner(config)

    # TOKENIZE
    # finetuner.tokenizer = AutoTokenizer.from_pretrained(config['BASE_MODEL_NAME'])
    tokenize = lambda batch: finetuner.tokenizer(batch['text'], padding='max_length', truncation=True)
    dataset_encoded = finetuner.dataset.map(tokenize, batched=True)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f'{config["BASE_MODEL_NAME"]}-finetuned-{config["DATASET_NAME"]}_{config["DATASET_CONFIG_NAME"]}',
        num_train_epochs=config['NUM_EPOCHS'],
        learning_rate=config['LEARNING_RATE'],
        per_device_train_batch_size=config['BATCH_SIZE'],
        per_device_eval_batch_size=config['BATCH_SIZE'],
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        disable_tqdm=False,
        logging_dir='./logs',
        push_to_hub=True,
        log_level=config['LOG_LEVEL'],
    )

    trainer = Trainer(
        model=finetuner.model,
        args=training_args,
        train_dataset=dataset_encoded['train'],
        eval_dataset=dataset_encoded[config['EVAL_DATASET']],
        compute_metrics=finetuner.compute_metrics_func,
    )
  
    print("Start TRAINING")
    trainer.train()
    trainer.evaluate()

    print("PUSH MODEL TO THE HUB")
    trainer.push_to_hub()
    logging.info("Model pushed to Hugging Face Hub.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning models with Hugging Face Transformers')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    args = parser.parse_args()
    main(args.config)