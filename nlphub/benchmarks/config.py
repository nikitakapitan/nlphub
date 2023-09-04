"""
This is config file for Performance Benchmark report generation

DATASET:
Any dataset is datasets.DatasetDict type with 'train', 'validation' and 'test'
Any dataset text-feature is supposed to always be 'text'
However, label-featuer is automatically parsed (ex. 'label', 'intent' etc)

"""
from nlphub import BERTClassificationBenchmark

example = {
    'optim_type' : 'BERT-base', # free user comment
    'metrics' : ['accuracy'], # list of metrics in str format
    
    'dataset' : { # dict that adapts the labels for particular dataset TBD
        'label' : None, # ex. 'label'
        'text' : None, # ex. 'intent'

    }
}

model_to_benchmark = {
    'classification' : {
        'BERT' : BERTClassificationBenchmark,

    },
    'ner' : {
        None
    },
    'question-answering' : {
        'miniLM' : None
    }
}