import pandas as pd
import matplotlib.pyplot as plt

def plot_char_histogram(dataset, split='train', text_regex = 'text', quantile=0.98):
    # dataset: DataDict

    ds = pd.DataFrame(dataset[split])

    ds['char_count'] = ds[text_regex].apply(len)

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


