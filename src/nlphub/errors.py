"""
Optional file to compute some metrics for error analysis
"""

import torch
from copy import deepcopy


def _forward_pass_with_label(batch, model, device, tokenizer):
    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}

    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = torch.nn.functional.cross_entropy(output.logits, batch['label'].to(device), reduction="none")

        return {"loss": loss.cpu().numpy(), 
                "predicted_label": pred_label.cpu().numpy()}

def error_analysis(dataset_encoded, model, device, tokenizer):

    dataset_encoded = deepcopy(dataset_encoded)

    def label_int2str(row):
        return data.features["label"].int2str(row)

    _fwd_pass = lambda batch : _forward_pass_with_label(batch, model=model, device=device, tokenizer=tokenizer)

    dataset_encoded.set_format("torch")
    dataset_encoded["validation"] = dataset_encoded["validation"].map(
        _fwd_pass, batched=True, batch_size=16)

    dataset_encoded.set_format("pandas")
    cols = ["text", "label", "predicted_label", "loss"]
    df_valid= dataset_encoded["validation"][:][cols]
    df_valid["label"] = df_valid["label"].apply(label_int2str)
    df_valid["predicted_label"] = (df_valid["predicted_label"]
                              .apply(label_int2str))

    return df_valid.sort_values("loss", ascending=False).head(10)