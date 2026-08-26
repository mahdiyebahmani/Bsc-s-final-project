import numpy as np

from hazm import word_tokenize
from src.utils import NEURAL_DATA_DIR
import fasttext

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def split_data(df, text_column):
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    print(
        f"{text_column}: "
        f"Train size: {len(X_train)}, "
        f"Test size: {len(X_test)}"
    )

    return X_train, X_test, y_train, y_test


def create_classical_features(X_train, X_test):
    vectorizer_configs = [
        ('bow', (1, 1), 'Unigram'),
        ('bow', (2, 2), 'Bigram'),
        ('bow', (1, 2), 'Unigram+Bigram'),
        ('tfidf', (1, 1), 'Unigram'),
        ('tfidf', (2, 2), 'Bigram'),
        ('tfidf', (1, 2), 'Unigram+Bigram')
    ]

    feature_sets = {}

    for vec_type, ngram_range, feat_name in vectorizer_configs:

        if vec_type == 'bow':
            vec = CountVectorizer(
                ngram_range=ngram_range,
                max_features=20000
            )
        else:
            vec = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=20000
            )
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        feature_sets[(vec_type, feat_name)] = (
            X_train_vec,
            X_test_vec
        )

        print(
            f"{vec_type.upper()} {feat_name}: "
            f"train {X_train_vec.shape}, "
            f"test {X_test_vec.shape}"
        )

    return feature_sets


def train_fasttext(X_train, corpus_file="data/processed/train_corpus.txt"):
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for text in X_train:
            tokens = word_tokenize(text)
            f.write(' '.join(tokens) + '\n')

    print("Training FastText from scratch on your data...")

    ft_model = fasttext.train_unsupervised(
        corpus_file,
        model='skipgram',
        dim=300,
        minCount=5,
        epoch=10,
        thread=4
    )

    print("Done.")

    return ft_model


def review_to_vector(text, ft_model):
    tokens = word_tokenize(text)

    if not tokens:
        return np.zeros(300)

    vectors = [
        ft_model.get_word_vector(t)
        for t in tokens
    ]

    return np.mean(vectors, axis=0)


def create_fasttext_features(X_train, X_test, ft_model):

    print("Converting train set...")

    X_train_ft = np.array([
        review_to_vector(text, ft_model)
        for text in X_train
    ])

    print("Converting test set...")

    X_test_ft = np.array([
        review_to_vector(text, ft_model)
        for text in X_test
    ])

    print("Train shape:", X_train_ft.shape)
    print("Test shape:", X_test_ft.shape)

    return X_train_ft, X_test_ft



def save_fasttext_features(
    X_train_ft,
    X_test_ft,
    y_train,
    y_test,
    output_dir=NEURAL_DATA_DIR
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_dir / "X_train_ft.npy",
        X_train_ft
    )

    np.save(
        output_dir / "X_test_ft.npy",
        X_test_ft
    )

    np.save(
        output_dir / "y_train.npy",
        y_train.to_numpy()
    )

    np.save(
        output_dir / "y_test.npy",
        y_test.to_numpy()
    )

    print(
        f"FastText features saved to:\n{output_dir}"
    )

def save_fasttext_model(ft_model, output_dir):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = output_dir / "fasttext.bin"

    ft_model.save_model(
        str(model_path)
    )

    print(f"FastText model saved to: {model_path}")