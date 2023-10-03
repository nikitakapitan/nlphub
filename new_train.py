# Colab:
# Turn ON GPU
# !git clone https://github.com/nikitakapitan/nlphub.git
# !mv nlphub/train.yaml .
# !mkdir logs
# !pip install datasets transformers evaluate accelerate 

# python train.py --config train.yaml
 
import os
import yaml
import logging
import argparse
import time
from transformers import TrainingArguments, Trainer
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
    tokenize = lambda batch: finetuner.tokenizer(batch['text'], truncation=True)
    dataset_encoded = finetuner.dataset.map(tokenize, batched=True)

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
        model=finetuner.model,
        args=training_args,
        train_dataset=dataset_encoded['train'],
        eval_dataset=dataset_encoded[config['EVAL_DATASET']],
        tokenizer=finetuner.tokenizer,
        compute_metrics=finetuner.compute_metrics_func,
    )

    # Train and Evaluate
    try:
        print("Start TRAINING")
        trainer.train()
        trainer.evaluate()
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        exit(1)

    # Push to Hub
    try:
        print("PUSH MODEL TO THE HUB")
        trainer.push_to_hub()
        logging.info("Model pushed to Hugging Face Hub.")
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning models with Hugging Face Transformers')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    args = parser.parse_args()
    main(args.config)