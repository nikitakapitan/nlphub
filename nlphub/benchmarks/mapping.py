from transformers import (
    AutoModelForSequenceClassification,

)

task_to_auto_model = {
    'text-classification' : AutoModelForSequenceClassification,
}