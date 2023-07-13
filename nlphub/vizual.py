import pandas as pd
from matplotlib import pyplot as plt
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from copy import deepcopy

from nlphub.hidden_state import prepare_data


def output_distribution(dataset, train=True):

    dataset = deepcopy(dataset)

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




def plot_umap(X_train, y_train, labels):

    X_scaled = MinMaxScaler().fit_transform(X_train)

    mapper = UMAP(n_components=2, metric='cosine').fit(X_scaled)

    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = y_train

    fig, axes = plt.subplots(2, 3, figsize=(7,5))
    axes = axes.flatten()
    cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]

    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_emb_sub = df_emb.query(f"label == {i}")
        axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                    gridsize=20, linewidths=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])

    plt.tight_layout()
    plt.show()
    

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


def plt_bars(preds, labels, title):
    preds_df = pd.DataFrame(preds[0])
    plt.bar(labels, 100 * preds_df["score"], color='C0')
    plt.title(f'"{title}"')
    plt.ylabel("Class probability (%)")
    plt.show()