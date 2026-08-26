import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)


class SentimentDataset:

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = str(
            self.texts.iloc[idx]
        )

        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    acc = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1
    }


class WeightedTrainer(Trainer):

    def __init__(
        self,
        class_weights=None,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(
                logits.device
            )
        )

        loss = loss_fct(
            logits,
            labels
        )

        if return_outputs:
            return loss, outputs

        return loss


def train_parsbert(
    X_train,
    X_test,
    y_train,
    y_test
):

    model_name = (
        "HooshvareLab/"
        "bert-fa-base-uncased"
    )

    device = torch.device("cpu")

    print(
        "Using device:",
        device
    )

    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    print(
        "\nTrain size     :",
        len(X_train_main)
    )

    print(
        "Validation size:",
        len(X_val)
    )

    print(
        "Test size      :",
        len(X_test)
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_main),
        y=y_train_main
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float
    )

    print(
        "Class weights:",
        class_weights.numpy()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_name,
            num_labels=3
        )
    )

    model.to(device)

    train_dataset = SentimentDataset(
        X_train_main,
        y_train_main,
        tokenizer,
        max_length=128
    )

    val_dataset = SentimentDataset(
        X_val,
        y_val,
        tokenizer,
        max_length=128
    )

    test_dataset = SentimentDataset(
        X_test,
        y_test,
        tokenizer,
        max_length=128
    )

    training_args = TrainingArguments(

        output_dir="./parsbert-sentiment-results",

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=50,

        num_train_epochs=2,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,

        learning_rate=2e-5,
        weight_decay=0.01,

        optim="adamw_torch",

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=1,

        report_to="none",

        use_cpu=True
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=1
            )
        ]
    )

    trainer.train()

    predictions = trainer.predict(
        test_dataset
    )

    pred_labels = np.argmax(
        predictions.predictions,
        axis=-1
    )

    print(
        "\n********  ParsBERT RESULTS  ********"
    )

    print(
        classification_report(
            y_test,
            pred_labels,
            target_names=[
                "not_recommended",
                "no_idea",
                "recommended"
            ],
            zero_division=0
        )
    )

    model.save_pretrained(
        "./models/parsbert"
    )

    tokenizer.save_pretrained(
        "./models/parsbert"
    )

    print(
        "Model and tokenizer saved."
    )

    return (
        trainer,
        model,
        tokenizer,
        pred_labels
    )