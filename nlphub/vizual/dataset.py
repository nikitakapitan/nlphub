import pandas as pd
import matplotlib.pyplot as plt

def plot_char_histogram(dataset, name, split='train', quantile=0.98):
    # dataset: DataDict

    # slice on particualr split.
    if 'train' in dataset or 'test' in dataset:
        dataset = dataset[split]
    
    preprocess_func = {
        None : lambda x : x,
        'narrativeqa' : preprocess_narrative_qa,
        'allenai/qasper' : preprocess_qasper,
                       }[name]

    df = preprocess_func(dataset)

    df['char_count'] = df['text'].apply(len)

    threshold = df['char_count'].quantile(quantile)
    threshold_df = df[df['char_count'] <= threshold]
    
    min_chars = df['char_count'].min()
    max_chars = df['char_count'].max()

    plt.figure(figsize=(10, 6))
    plt.hist(threshold_df['char_count'], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of Character Counts in Dataset[{split}]')
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

    texts, summaries, questions, answers = [], [], [], [] 

    for example in dataset:
        texts.append(example['document']['text'])
        summaries.append(example['document']['summary']['text'])
        questions.append(example['question']['text'])
        one_question_answers = []
        for ans in example['answers']:
            one_question_answers.append(ans['text'])
        answers.append(one_question_answers)

    df = pd.DataFrame({'text': texts, 
                        'summary' : summaries, 
                        'question': questions,
                        'answers' : answers})
    return df
    

def preprocess_qasper(ds):
    None