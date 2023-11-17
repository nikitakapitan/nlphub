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
    'glue': ['cola', 'sst2'],
    'clinc_oos': ['plus', None],
    'imdb': [None],
}
model_options = ['bert-base-uncased', 'distilbert-base-uncased']

# Create widgets
task_widget = widgets.Dropdown(options=task_options, value=data['TASK'], description='TASK:')
base_model_name_widget = widgets.Dropdown(options=model_options, value=data['BASE_MODEL_NAME'], description='MODEL:')
dataset_name_widget = widgets.Dropdown(options=dataset_name_options, value=data['DATASET_NAME'], description='DATASET:')
dataset_config_name_widget = widgets.Dropdown(options=dataset_config_name_options[data['DATASET_NAME']], value=data['DATASET_CONFIG_NAME'], description='DATA CFG:')


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

# Button to save changes
save_button = widgets.Button(description="Save Changes")

# Save function
def save_changes(b):
    data['TASK'] = task_widget.value
    data['DATASET_NAME'] = dataset_name_widget.value
    data['DATASET_CONFIG_NAME'] = dataset_config_name_widget.value
    data['BASE_MODEL_NAME'] = base_model_name_widget.value
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
    display(task_widget, base_model_name_widget, dataset_name_widget, dataset_config_name_widget, save_button, output)