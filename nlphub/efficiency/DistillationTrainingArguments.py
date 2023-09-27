from transformers import TrainingArguments

class DistillationTrainingArguments(TrainingArguments):

    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):

        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature