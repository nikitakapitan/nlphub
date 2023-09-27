
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
            'clinc_oos' : "transfromersbook/bert-base-uncased-finetuned-clinc",
            'emotion' : "nikitakapitan/bert-base-uncased-finetuned-emotion",
            'imdb' : "nikitakapitan/bert-base-uncased-finetuned-imdb",
        },
        'distilbert-base-uncased' : {
            'clinc_oos' : "nikitakapitan/distilbert-base-uncased-finetuned-clinc_oos",
            'emotion' : "nikitakapitan/distilbert-base-uncased-finetuned-emotion", 
            'imdb' : "nikitakapitan/distilbert-base-uncased-finetuned-imdb",
        }
    },

    'question-answering' : {
        'bert-base-uncased' : {
            'squad?' : None
        }
    }
}
