import pandas as padding

def output_distribution(dataset, train=True):

    data = dataset['train'] if train else dataset['test']
    data.set_format(type='pandas')
    df = data[:]

    def label_int2str(row):
        return data.features["label"].int2str(row)
    
    builder_name = data.builder_name
    df[builder_name] = data["label"].apply(label_int2str)
    df[builder_name].value_counts(ascending=True).plot.barh()

    df['n_words'] = df['text'].str.split().apply(len)
    df.boxplot('n_words', by=builder_name, showfliers=False)





