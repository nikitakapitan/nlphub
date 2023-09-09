from transformers import (
    AutoModelForAudioClassification,

)

task_to_auto_model = {
    'text-classification' : AutoModelForAudioClassification,
}