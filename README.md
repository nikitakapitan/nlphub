# NLP Hub: Fine-Tuning and Distillation of LLM Models

Welcome to NLP Hub, a comprehensive toolkit for fine-tuning and distilling Large Language Models (LLMs) with ease. This project allows AI researchers and enthusiasts to utilize the power of Google Colab's GPU resources and Hugging Face's model hub to enhance and tailor language models for specific tasks.

## Quick Start

Utilize your Hugging Face account and Google Colab's free GPU-enabled environment to start fine-tuning your LLM models in just a few steps.

### Prerequisites

- Google Colab account
- Hugging Face account

### Installation

1. **Clone the Repository**
   
   ```bash
   !git clone https://github.com/nikitakapitan/nlphub.git
   !mv nlphub/finetune.yaml .
   !mkdir logs

2. **Install Dependencies**
   
   ```bash
   !pip install datasets transformers evaluate accelerate 

3. **Hugging Face Login**

   ```bash
   from huggingface_hub import login
   login("hf_YOUR_TOKEN_HERE")

4. **Configuration Setup**
Import the configuration module and set up your fine-tuning parameters using an interactive widget.

   ```bash
   from nlphub.vizual.colab_yaml import config_yaml
   config_yaml() 

## Fine-Tuning Your Model

Once you have configured your settings:

  ```bash
  !python nlphub/finetune.py --config finetune.yaml
  ```

This command will start the fine-tuning process based on your specified configurations. Upon completion, the new model will automatically appear on your Hugging Face account.

## Contributing
We welcome contributions! If you'd like to improve or add features to NLP Hub, please feel free to submit a pull request.
