
from nlphub.benchmarks.tasks.ClassificationBenchmark import (
    ClassificationBenchmark
)

task_to_benchmark = {
    'text-classification'  : ClassificationBenchmark,
    'question-answering'  : None,
    'ner'  : None,
}

task_model_dataset_to_ft_model = {
    'text-classification'  : {
        'bert-base-uncased' : {
            'clinc' : "transfromersbook/bert-base-uncased-finetuned-clinc",
            'emotions' : "nikitakapitan/bert-base-uncased-finetuned-emotion",
            'imdb' : "nikitakapitan/bert-base-uncased-finetuned-imdb",
        },
        'distilbert-base-uncased' : {
            'clinc' : "nikitakapitan/distilbert-base-uncased-finetuned-clinc_oos",
            'emotions' : "nikitakapitan/distilbert-base-uncased-finetuned-emotion", 
            'imdb' : "nikitakapitan/distilbert-base-uncased-finetuned-imdb",
        }
    },

    'question-answering' : {
        'bert-base-uncased' : {
            'squad?' : None
        }
    }
}
