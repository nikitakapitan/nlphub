from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer)

from nlphub import (
    ClassificationBenchmark,
    NERBenchmark,
    QABenchmark,
)

from datasets import load_dataset

# mapping from task name to Hugging Face AutoModel
task_to_automodel = {
        'classification': AutoModelForSequenceClassification,
        'ner': AutoModelForTokenClassification,
        'question-answering': AutoModelForQuestionAnswering,
    }

task_to_benchmark = {
        'classification' : ClassificationBenchmark,
        'ner' : NERBenchmark,
        'quesntion-answering' : QABenchmark,
    }

def generate_report(task='classification'):

    models = ['bert-base-uncased']
    datasets = ['clinc']

    report = {}
    
    
    AutoModelClass = task_to_automodel[task]
    BenchmarkClass = task_to_benchmark[task]

    for model_name in models:
        benchmark = BenchmarkClass(model, tokenizer, dataset)

        model = AutoModelClass.from_pretrained(model_name)
        tokenizer = AutoTokenizer(model_name)
        report[model_name] = {}

        for dataset_name in datasets:
            dataset = load_dataset(dataset_name)
            result = benchmark.compute_performance(dataset)

            report[model_name][dataset_name] = result
        
        report[model_name].update(benchmark.compute_time())
        report[model_name].update(benchmark.compute_size())