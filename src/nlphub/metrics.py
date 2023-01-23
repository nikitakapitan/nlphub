from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    "pred: EvalPrediction object"

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {'accuracy' : accuracy, 'f1' : f1}