from datasets import load_dataset
import os

def get_datasetdict_size(dataset):
    # Initialize total size
    total_size = 0

    # Iterate over each split in the DatasetDict
    for split in dataset_dict.keys():
        # Get the dataset directory for the current split
        dataset_dir = dataset_dict[split].cache_files[0]['filename'].rsplit('/', 1)[0]

        # Calculate the size for the current split
        for dirpath, dirnames, filenames in os.walk(dataset_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

    # Convert size to megabytes
    size_in_mb = total_size / (1024 * 1024)
    print(f"The total size of all splits in the dataset is {size_in_mb:.2f} MB")