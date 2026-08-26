import numpy as np
import tensorflow as tf
import fasttext

from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import (
    compute_sample_weight,
    compute_class_weight
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    GRU,
    Dense,
    Dropout,
    Input,
    Bidirectional
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def train_mlp_fasttext(
    X_train_ft,
    X_test_ft,
    y_train,
    y_test,
    ft_model,
    evaluate_function
):

    sample_w = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )

    mlp.fit(
        X_train_ft,
        y_train,
        sample_weight=sample_w
    )

    pred = mlp.predict(X_test_ft)

    print(
        "\nMLP with Self-trained FastText Embeddings"
    )

    metrics = evaluate_function(
        y_test,
        pred
    )

    print(
        "Accuracy:",
        metrics['Accuracy']
    )

    print(
        "Macro F1:",
        metrics['Macro F1']
    )

    print(metrics['Report'])

    return mlp, pred, metrics


def prepare_sequences(
    X_train,
    X_test,
    ft_model,
    num_words=20000,
    max_len=100
):

    tokenizer = Tokenizer(
        num_words=num_words
    )

    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(
        X_train
    )

    X_test_seq = tokenizer.texts_to_sequences(
        X_test
    )

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=max_len,
        padding='post'
    )

    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=max_len,
        padding='post'
    )

    vocab_size = min(
        num_words,
        len(tokenizer.word_index) + 1
    )

    embedding_dim = 300

    embedding_matrix = np.zeros(
        (vocab_size, embedding_dim)
    )

    for word, i in tokenizer.word_index.items():

        if i < vocab_size:

            embedding_matrix[i] = (
                ft_model.get_word_vector(word)
            )

    return (
        tokenizer,
        X_train_pad,
        X_test_pad,
        vocab_size,
        embedding_dim,
        embedding_matrix
    )


def create_class_weights(y_train):

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    return dict(
        enumerate(class_weights)
    )


def train_cnn(
    vocab_size,
    embedding_dim,
    embedding_matrix,
    X_train_pad,
    X_test_pad,
    y_train,
    y_test,
    class_weight_dict,
    evaluate_function
):

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    cnn_model = Sequential([
        Input(shape=(100,)),

        Embedding(
            vocab_size,
            embedding_dim,
            weights=[embedding_matrix],
            trainable=True
        ),

        Conv1D(
            128,
            5,
            activation='relu'
        ),

        GlobalMaxPooling1D(),

        Dense(
            64,
            activation='relu'
        ),

        Dropout(0.6),

        Dense(
            3,
            activation='softmax'
        )
    ])

    cnn_model.compile(
        optimizer=Adam(
            learning_rate=0.0005
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    cnn_model.summary()

    history_cnn = cnn_model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )

    cnn_pred = np.argmax(
        cnn_model.predict(X_test_pad),
        axis=1
    )

    metrics = evaluate_function(
        y_test,
        cnn_pred
    )

    print(
        "\n********  CNN RESULTS  **********"
    )

    print(metrics['Report'])

    return cnn_model, history_cnn, cnn_pred, metrics


def train_rnn(
    vocab_size,
    embedding_dim,
    embedding_matrix,
    X_train_pad,
    X_test_pad,
    y_train,
    y_test,
    class_weight_dict,
    evaluate_function
):

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    rnn_model = Sequential([
        Input(shape=(100,)),

        Embedding(
            vocab_size,
            embedding_dim,
            weights=[embedding_matrix],
            trainable=True
        ),

        Bidirectional(
            GRU(
                128,
                dropout=0.3
            )
        ),

        Dense(
            64,
            activation='relu'
        ),

        Dropout(0.5),

        Dense(
            3,
            activation='softmax'
        )
    ])

    rnn_model.compile(
        optimizer=Adam(
            learning_rate=0.0005,
            clipnorm=1.0
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    rnn_model.summary()

    history_rnn = rnn_model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )

    rnn_pred = np.argmax(
        rnn_model.predict(X_test_pad),
        axis=1
    )

    metrics = evaluate_function(
        y_test,
        rnn_pred
    )

    print(
        "\n********  RNN RESULTS  ********"
    )

    print(metrics['Report'])

    return rnn_model, history_rnn, rnn_pred, metrics
    