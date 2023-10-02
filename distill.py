# Colab:
# Turn ON GPU
# !git clone https://github.com/nikitakapitan/nlphub.git
# !mv nlphub/distill.yaml .
# !mkdir logs
# !pip install datasets transformers evaluate accelerate 

# python distill.py --config distill.yaml
 
import os
import yaml
import argparse
import logging
import time


from datasets import load_dataset
import evaluate

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from nlphub import  DistillationTrainingArguments, DistillationTrainer
from nlphub.utils import rename_split_label_key, get_dataset_num_classes

# Initialize logging
if not os.path.exists('/content/logs/'):
    os.makedirs('/content/logs/')
logging.basicConfig(filename=f"logs/train_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log", level=logging.INFO)


def main(args):

    with open(args, 'r') as f:
        config = yaml.safe_load(f)

    # Log input configs
    logging.info("Input Configurations:")
    logging.info(yaml.dump(config))

    os.environ['TRANSFORMERS_TOKEN'] = config['HF_TOKEN']
    device = 'cuda'

    # Load Dataset
    try:
        dataset_config_name = config.get('DATASET_CONFIG_NAME') # can be None
        dataset = load_dataset(config['DATASET_NAME'], dataset_config_name)
        for split in dataset:
            dataset[split] = rename_split_label_key(dataset[split])
        num_classes = get_dataset_num_classes(dataset['train'].features)
        logging.info(f"Dataset {config['DATASET_NAME']} loaded. {num_classes=}")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        exit(1)

    # INIT Tokenization
    try:
        student_tokenizer = AutoTokenizer.from_pretrained(config['STUDENT'])
        logging.info(f"Tokenizer for {config['STUDENT']} initialized.")
    except Exception as e:
        logging.error(f"Error initializing tokenizer: {e}")
        exit(1)

    # Dynamic Class Mapping
    AutoModelClass = {
    'text-classification' : AutoModelForSequenceClassification
    }[config['TASK']]
    logging.info(f"Mapped to AutoModel Class: {AutoModelClass.__name__}")

    # INIT Model (Teacher)
    try:
        teacher = AutoModelClass.from_pretrained(config['BASE_MODEL_NAME'], num_labels=num_classes)
        logging.info(f"Model {teacher.__class__.__name__} initialized with {num_classes} classes.")
        teacher.to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit(1)

    # INIT Model (Student)
    def student_init():
        student_config = AutoConfig.from_pretrained(config['STUDENT'],
                                                    num_labels=num_classes,
                                                    id2label=teacher.config.id2label,
                                                    label2id=teacher.config.label2id)
        return AutoModelClass.from_pretrained(config['STUDENT'], config=student_config).to(device)


    # # DEFINE compute_metrics
    def compute_eval_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        accuracy_score = evaluate.load_metric("accuracy")
        return  accuracy_score.compute(predictions=preds, references=labels)

    # Tokenize
    tokenize = lambda batch: student_tokenizer(batch['text'], truncation=True)
    dataset_encoded = dataset.map(tokenize, batched=True)

    # Distill Training Arguments
    student_training_args =  DistillationTrainingArguments(
        output_dir=f'{config["TEACHER"]}-distilled-{config["DATASET_NAME"]}',
        num_train_epochs=config['NUM_EPOCHS'],
        learning_rate=config['LEARNING_RATE'],
        per_device_train_batch_size=config['BATCH_SIZE'],
        per_device_eval_batch_size=config['BATCH_SIZE'],
        alpha = config['ALPHA'],
        weight_decay=0.01,
        evaluation_strategy='epoch',
        disable_tqdm=False,
        logging_dir='./logs',
        push_to_hub=True,
        log_level=config['LOG_LEVEL'],
    )

    trainer = DistillationTrainer(
        model_init=student_init,
        teacher_model=teacher,
        args=student_training_args,
        train_dataset=dataset_encoded['train'],
        eval_dataset=dataset_encoded[config['EVAL_DATASET']],
        tokenizer=student_tokenizer,
        compute_metrics=compute_eval_metrics,
    )

    # Train and Evaluate
    try:
        print("Start TRAINING")
        logging.info("Start TRAINING")
        trainer.train()
        trainer.evaluate()
        logging.info("End TRAINING")
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
    parser = argparse.ArgumentParser(description='Distilling models with Hugging Face Transformers')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    args = parser.parse_args()
    main(args.config)