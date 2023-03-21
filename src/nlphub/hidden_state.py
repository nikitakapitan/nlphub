
import torch
import numpy as np

def get_hidden_state(data_encoded, model, tokenizer, device=None):

    if device == None:
      device = 'cpu'

    extract_hidden_state_model = lambda batch : _extract_hidden_state(batch, model=model, tokenizer=tokenizer, device=device)

    data_encoded.set_format("torch")
    data_hidden = data_encoded.map(extract_hidden_state_model, batched=True)
    return data_hidden

def _extract_hidden_state(batch, model, tokenizer, device):

  inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}

  with torch.no_grad():
    last_hidden_state = model(**inputs).last_hidden_state

  # return THE FIRST token aka. [CLS]
  return {"hidden_state" : last_hidden_state[:,0].cpu().numpy()}

  
def prepare_data(data_hidden):

  X_train = np.array(data_hidden['train']['hidden_state'])
  X_valid = np.array(data_hidden['validation']['hidden_state'])
  y_train = np.array(data_hidden['train']['label'])
  y_valid = np.array(data_hidden['validation']['label'])

  return X_train, X_valid, y_train, y_valid
