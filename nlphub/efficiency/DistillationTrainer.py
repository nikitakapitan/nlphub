import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class DistillationTrainer(Trainer):

    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        
        # return_outputs = True : (loss, logits) | = False : (loss)
        output_student = model(**inputs)

        loss_ce = output_student.loss
        logits_student = output_student.logits

        with torch.no_grad():
            output_teacher = self.teacher_model(**inputs)
            logits_teacher = output_teacher.logits

        loss_fct = nn.KLDivLoss(reduction="batchmean")

        # nn.KLDivLoss official doc: 
        # to avoid underflow y_pred is expected log-prob, y_target - normal prob unless log_target=True
        loss_kd = self.args.temperature ** 2 * loss_fct(
            F.log_softmax(logits_student / self.args.temperature, dim=-1),
            F.softmax(logits_teacher / self.args.temperature, dim=-1))
        
        total_loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd
        return (total_loss, output_student) if return_outputs else total_loss
        