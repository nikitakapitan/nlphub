import pandas as pd
import matplotlib.pyplot as plt

def plot_char_histogram(dataset, name=None, split='train', quantile=0.98):
    # dataset: DataDict
    
    preprocess_func = {
        None : lambda x : x,
        'narrativeqa' : preprocess_narrative_qa,
        'allenai/qasper' : preprocess_qasper,
                       }[name]

    ds = preprocess_func(dataset[split])

    ds = pd.DataFrame(ds)

    ds['char_count'] = ds['text'].apply(len)

    threshold = ds['char_count'].quantile(quantile)
    sliced_ds = ds[ds['char_count'] <= threshold]
    
    min_chars = ds['char_count'].min()
    max_chars = ds['char_count'].max()

    plt.figure(figsize=(10, 6))
    plt.hist(sliced_ds['char_count'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Character Counts in Dataset')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')

    # Displaying min and max values
    plt.text(x=threshold/max_chars*max_chars, y=0, s=f"Max: {max_chars}", color='red')
    plt.text(x=min_chars, y=0, s=f"Min: {min_chars}", color='green')

    plt.show()


def preprocess_narrative_qa(dataset):
    """return dict with keys: 
    {'text' : str 
    'summary' : str
    'question' : str
    'answers' : List[str]}
    """
    ds = {}
    ds['text'] = dataset['document']['text']
    ds['summary'] = dataset['document']['summary']['text']
    ds['question'] = dataset['document']['question']['text']
    ds['answers'] = []
    for ans in dataset['document']['answers']:
        ds['answers'].append(ans['text'])
    

def preprocess_qasper(ds):
    None