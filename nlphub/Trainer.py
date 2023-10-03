from abc import ABC, abstractclassmethod

import os
import yaml
import argparse
import logging
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import evaluate
from nlphub.utils import rename_split_label_key, get_dataset_num_classes

class Trainer(ABC):

    def __init__(self, config):

        self.config = config
        self.setup()            # logging, self.AutoModelClass, self.device
        self.load_dataset()     # self.dataset, self.num_classes
        self.init_tokenizer()   # self.tokenizer
        self.init_model()       # self.model
        self.define_compute_metrics()  # self.compute_metrics_func

    def setup(self):
        logging.info("Input Configurations:")
        logging.info(yaml.dump(self.config))

        # Task To AutoClass:
        self.AutoModelClass = {
        'text-classification' : AutoModelForSequenceClassification
        }[self.config['TASK']]

        self.device = 'cuda'

    def load_dataset(self):
        dataset_config_name = self.config.get('DATASET_CONFIG_NAME') # can be None
        dataset = load_dataset(self.config['DATASET_NAME'], dataset_config_name)
        for split in dataset:
            dataset[split] = rename_split_label_key(dataset[split])
        num_classes = get_dataset_num_classes(dataset['train'].features)
        logging.info(f"Dataset {self.config['DATASET_NAME']} loaded ✅ {num_classes=}")


        self.dataset = dataset
        self.num_classes = num_classes

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config['BASE_MODEL_NAME'])
        logging.info(f"INIT Token for {self.config['BASE_MODEL_NAME']}: initialized ✅")
        self.tokenizer = tokenizer

    def init_model(self):
        model = self.AutoModelClass.from_pretrained(self.config['BASE_MODEL_NAME'], num_labels=self.num_classes)
        logging.info(f"INIT Model: {model.__class__.__name__} initialized with {self.num_classes} classes ✅")
        model.to(self.device)
        self.model = model


    def define_compute_metrics(self):
        # abstract method by default uses accuracy.

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            accuracy_score = evaluate.load("accuracy")
            return  accuracy_score.compute(predictions=preds, references=labels)

        self.compute_metrics_func = compute_metrics # function