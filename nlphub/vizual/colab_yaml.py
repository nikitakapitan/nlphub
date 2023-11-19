import yaml
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load your YAML file
yaml_file = 'train.yaml'  # Replace with your YAML file path

with open(yaml_file) as file:
    data = yaml.safe_load(file)

# Define the options for the widgets
task_options = ['text-classification', 'ner', 'question-answering']
dataset_name_options = ['imdb', 'glue', 'clinc_oos']
dataset_config_name_options = {
    'glue': ['cola', 'sst2', 'mrpc', 'mnli', 'qnli', 'qqp', 'rte', 'stsb', 'wnli', 'ax'],
    'clinc_oos': ['plus', None],
    'imdb': [None],
}
model_options = ['bert-base-uncased', 'distilbert-base-uncased']

# Create widgets
task_widget = widgets.Dropdown(options=task_options, value=data['TASK'], description='TASK:')
base_model_name_widget = widgets.Dropdown(options=model_options, value=data['BASE_MODEL_NAME'], description='MODEL:')
dataset_name_widget = widgets.Dropdown(options=dataset_name_options, value=data['DATASET_NAME'], description='DATASET:')
dataset_config_name_widget = widgets.Dropdown(options=dataset_config_name_options[data['DATASET_NAME']], value=data['DATASET_CONFIG_NAME'], description='DATA CFG:')

# Additional options for advanced settings
num_epochs_options = [1, 3, 10, 100]
weight_decay_options = [0.0, 0.01, 0.02, 0.03, 0.05]
learning_rate_options = [0.00001, 0.00002, 0.0001, 0.0002, 0.001]
push_to_hub_options = [True, False]
evaluation_strategy_options = ['no', 'steps', 'epoch']
batch_size_options = [8, 16, 32, 64]

# Advanced settings widgets
advanced_checkbox = widgets.Checkbox(value=False, description='Advanced')
batch_size_widget = widgets.Dropdown(options=batch_size_options, value=data.get('BATCH_SIZE', 16), description='BATCH SIZE:')
num_epochs_widget = widgets.Dropdown(options=num_epochs_options, value=data.get('NUM_EPOCHS', 1), description='EPOCHS:')
weight_decay_widget = widgets.Dropdown(options=weight_decay_options, value=data.get('WEIGHT_DECAY', 0.0), description='W8 DECAY:')
learning_rate_widget = widgets.Dropdown(options=learning_rate_options, value=data.get('LEARNING_RATE', 0.0001), description='LRATE:')
push_to_hub_widget = widgets.Dropdown(options=push_to_hub_options, value=data.get('PUSH_TO_HUB', False), description='PUSH2HUB:')
evaluation_strategy_widget = widgets.Dropdown(options=evaluation_strategy_options, value=data.get('EVALUATION_STRATEGY', 'epoch'), description='EVAL STRAT:')

# Function to toggle advanced settings
def toggle_advanced_settings(change):
    if advanced_checkbox.value:
        display(num_epochs_widget, batch_size_widget, weight_decay_widget, learning_rate_widget, push_to_hub_widget, evaluation_strategy_widget)
    else:
        clear_output()
        config_yaml()

# Update function for DATASET_NAME change
def update_dataset_config_name_options(*args):
    dataset_config_name_widget.options = dataset_config_name_options[dataset_name_widget.value]
    update_metrics()

# Update METRICS based on DATASET_NAME
def update_metrics():
    if dataset_name_widget.value == 'glue':
        data['METRICS'] = [
            {'accuracy': {}},
            {'f1': {'average': 'weighted'}},
            {'glue': [dataset_config_name_widget.value]}
        ]
    else:
        data['METRICS'] = [
            {'accuracy': {}},
            {'f1': {'average': 'weighted'}}
        ]

# Observe changes in DATASET_NAME
dataset_name_widget.observe(update_dataset_config_name_options, 'value')

advanced_checkbox.observe(toggle_advanced_settings, 'value')

# Button to save changes
save_button = widgets.Button(description="Save Changes")

# Save function
def save_changes(b):
    data['TASK'] = task_widget.value
    data['DATASET_NAME'] = dataset_name_widget.value
    data['DATASET_CONFIG_NAME'] = dataset_config_name_widget.value
    data['BASE_MODEL_NAME'] = base_model_name_widget.value
    if advanced_checkbox.value:
        data['NUM_EPOCHS'] = num_epochs_widget.value
        data['BATCH_SIZE'] = batch_size_widget.value
        data['WEIGHT_DECAY'] = weight_decay_widget.value
        data['LEARNING_RATE'] = learning_rate_widget.value
        data['PUSH_TO_HUB'] = push_to_hub_widget.value
        data['EVALUATION_STRATEGY'] = evaluation_strategy_widget.value

    update_metrics()

    with open(yaml_file, 'w') as file:
        yaml.safe_dump(data, file)

    with output:
        clear_output()
        print("YAML file updated!")

save_button.on_click(save_changes)

# Output
output = widgets.Output()

# Display widgets
def config_yaml():
    display(task_widget, base_model_name_widget, dataset_name_widget, dataset_config_name_widget, advanced_checkbox, save_button, output)
    
    if advanced_checkbox.value:
        display(num_epochs_widget, weight_decay_widget, learning_rate_widget, push_to_hub_widget, evaluation_strategy_widget)
