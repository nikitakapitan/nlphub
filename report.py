# python report.py --config report.yaml

import os
import yaml
import json
import argparse
import logging
import time

import torch
from datasets import load_dataset
from transformers import pipeline

from nlphub import PerformanceBenchmark, ClassificationBenchmark
from nlphub.benchmarks.mapping import task_model_dataset_to_ft_model, task_to_benchmark

device = 'cuda'

# Initialize logging
if not os.path.exists('/content/logs/'):
    os.makedirs('/content/logs/')
logging.basicConfig(filename=f"logs/report_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log", level=logging.INFO)


def main(config_path):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logging.info("Input Configurations:")
    logging.info(yaml.dump(config))

    report = {}

    for dataset_name in config['DATASET_NAMES']:
        logging.info(f"Loading {dataset_name} dataset ...")
        dataset = load_dataset(dataset_name, split='test')

        for model_name in config['MODEL_NAMES']:
            logging.info(f"Loading {model_name} pipeline ...")

            # truncation : crop input text to model max_length | device=0 : first available GPU
            pipe = pipeline(config['TASK'], model=model_name, truncation=True, device=0) 
            

            BenchmarkClass = task_to_benchmark[config['TASK']]
            benchmark = BenchmarkClass(pipe, dataset, config['METRICS'])
            metrics = benchmark.run_benchmark()

            if dataset_name not in report:
                report[dataset_name] = {}
            report[dataset_name][model_name] = metrics

            with open('results.json', 'w') as f:
                results = {}
                results['report'] = report
                results['config'] = config
                json.dump(results, f, indent=4)

    logging.info(f"Benchmark Report: {report}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance benchmark')
    parser.add_argument('--config', type=str, required=True, help='Path to the Yaml config file')
    args = parser.parse_args()
    main(args.config)