import pickle
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from models import ArticleDB
from nlp import remove_number
from utils import load_yaml

import tensorflow as tf
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import datasets
from sklearn.preprocessing import LabelEncoder


def load_data():
    db = ArticleDB()
    label_names = load_yaml("labels")
    session = db.create_session()

    data = db.get_articles_by_specific_categories(session, label_names)
    df = pd.DataFrame(data, columns=["id", "category", "text"])

    return df


def svm_straing(X_train, y_train):
    print("start training")

    text_clf = Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(
                    max_features=50000,
                    lowercase=False,
                    min_df=1,
                    stop_words=None,
                    preprocessor=remove_number,
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )
    text_clf = text_clf.fit(X_train, y_train)

    return text_clf


class PhobertTraining():
    def __init__(self, phobert_base, model=None) -> None:
        self.phobert_base = phobert_base
        self.model = model

    def make_data(self, path, label_encoder, test_size=0.15, batch_size=8):
        # Load data to Datasaet object
        train_data = datasets.load_dataset(
            'csv',
            data_files=path
        )['train']

        # Tokenizer text to input_ids, attention_mask, token_type_ids
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.phobert_base, model_max_length=512)

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)
        tokenized_datasets = train_data.map(tokenize_function, batched=True)

        # Split data to train and test set
        train_data = tokenized_datasets.train_test_split(
            test_size=test_size, shuffle=True)
        train_dataset = train_data['train']
        eval_dataset = train_data['test']

        # Encode label to integer
        self.le = label_encoder
        train_labels = self.le.transform(train_dataset['category'])
        eval_labels = self.le.transform(eval_dataset["category"])

        # Convert to batch tensor
        tf_train_dataset = train_dataset.with_format("tensorflow")
        tf_eval_dataset = eval_dataset.with_format("tensorflow")

        train_features = {
            x: tf_train_dataset[x]for x in self.tokenizer.model_input_names
        }
        train_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (train_features, train_labels)
        )
        self._train_tf_dataset = train_tf_dataset.shuffle(len(tf_train_dataset))#.batch(batch_size)

        eval_features = {
            x: tf_eval_dataset[x] for x in self.tokenizer.model_input_names
        }
        eval_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (eval_features, eval_labels)
        )
        self._eval_tf_dataset = eval_tf_dataset#.batch(batch_size)
        
        self.get_batch_data(batch_size)
        
    def get_batch_data(self, batch_size):
        self.train_tf_dataset = self._train_tf_dataset.batch(batch_size)
        self.eval_tf_dataset = self._train_tf_dataset.batch(batch_size)
        
        print(f'batch size = {batch_size}')
        print(f'train dataset shape: {self.train_tf_dataset}')
        print(f'validation dataset shape: {self.eval_tf_dataset}')
        

    def plot_train_history(self, history):
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        # summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        print('Load model successfully!')
        
    def load_weights(self, path):
        assert self.model is not None, "Init model before load weights"
        self.model.load_weights(path)

    def init_model(self, lr=5e-5, freeze=True):
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.phobert_base,
            num_labels=len(self.le.classes_)
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=tf.metrics.SparseCategoricalAccuracy(),
        )
        if freeze:
            self.model.layers[0].trainable = False

        print('init model successfully!')

    def fit(
        self,
        n_epochs=10,
        log_dir="./logs",
        checkpoint_dir="data/models/"
    ):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    checkpoint_dir, "model.{epoch:02d}-{val_loss:.2f}.h5"
                ),
                save_weights_only=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        ]

        assert self.model is not None, "Init model before training"

        print(self.model.summary())
        # fit the model
        history = self.model.fit(
            self.train_tf_dataset,
            epochs=n_epochs,
            verbose=1,
            validation_data=self.eval_tf_dataset,
            callbacks=callbacks,
        )

        # evaluate the model
        loss, accuracy = self.model.evaluate(self.eval_tf_dataset, verbose=1)
        print("Accuracy: %f" % (accuracy * 100))
        print(f"Loss: {loss}")

        return history
