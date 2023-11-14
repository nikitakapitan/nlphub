# %%capture
# !git clone https://github.com/nikitakapitan/nlphub.git
# !pip install datasets transformers evaluate accelerate 
# !mv nlphub/report.yaml .

# >>> Customize report.yaml

#!python nlphub/report.py --config report.yaml

"""This script produce a report for every model listed X for every dataset listed.

Pseudo-code.
FOR DATASET in DATASETS:
    FOR MODEL IN MODELS:
        create PIPE(TASK, MODEL)
        create BENCHMARK(PIPE, DATASET)
        run BENCHMAR.RUN()
        return METRICS (results.json)
"""

import os
import yaml
import json
import argparse
import logging
import time

from datasets import load_dataset
from transformers import pipeline

from nlphub import ClassificationBenchmark
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

        if dataset_name=='clinc_oos':
            dataset = load_dataset(dataset_name, 'plus', split='test')
        else:
            dataset = load_dataset(dataset_name, split='test')

        for model_name in config['MODEL_NAMES']:

            # truncation : crop input text to model max_length | device=0 : first available GPU
            pipe = pipeline(config['TASK'], model=model_name, truncation=True, device=0) 

            BenchmarkClass = {
                'text-classification'   : ClassificationBenchmark,
                'question-answering'    : None,
                'ner'                   : None,
            }[config['TASK']]

            metrics_config = {k:v for d in config['METRICS'] for k, v in d.items()} # cast list[dict] -> dict
            benchmark = BenchmarkClass(pipe, dataset, metrics_config)
            metrics = benchmark.run_benchmark()

            if dataset_name not in report:
                report[dataset_name] = {}
            report[dataset_name][model_name] = metrics

            with open('results.json', 'w') as f:
                results = {}
                results['report'] = report
                results['config'] = config
                json.dump(results, f, indent=4)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance benchmark')
    parser.add_argument('--config', type=str, required=True, help='Path to the Yaml config file')
    args = parser.parse_args()
    main(args.config)