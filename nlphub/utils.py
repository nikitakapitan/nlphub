import datasets

def rename_split_label_key(dataset) -> None:
    "dataset : transformers.Dataset (ex. dataset=DatasetDict['test'])"
    "rename given dataset split key ('intent', 'answer' etc) to 'label"
    for key, feature in dataset.features.items():
        if isinstance(feature, datasets.ClassLabel):
            current_key = key
    if current_key != 'label':
        dataset = dataset.rename_column(current_key, 'label')


def get_dataset_num_classes(features) -> int:
    "Extract num_classes from provided dataset"
    for key, feature in features.items():
        if hasattr(feature, 'num_classes'):
            return features[key].num_classes
    raise ValueError("Could not find a suitable label key with num_classes in the dataset features ")