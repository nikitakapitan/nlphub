import datasets

def rename_dataset_label_key(dataset) -> None:
    "rename any dataset key ('intent', 'answer' etc) to 'label"
    for key, feature in dataset['train'].features.items():
        if isinstance(feature, datasets.ClassLabel):
            current_key = key
    if current_key != 'label':
        for split in dataset.keys():
            dataset[split] = dataset[split].rename_column(current_key, 'label')


def get_dataset_num_classes(features) -> int:
    "Extract num_classes from provided dataset"
    for key, feature in features.items():
        if hasattr(feature, 'num_classes'):
            return features[key].num_classes
    raise ValueError("Could not find a suitable label key with num_classes in the dataset features ")