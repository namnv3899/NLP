from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

from phobert.dataloader import DataGenerator
from utils import load_yaml

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

MAX_LEN = load_yaml("max_seq_len")


def load_text(data, ids_method):
    print("Do dai du lieu la ", len(data))
    ids = []
    for i in tqdm(range(len(data)), desc="Text to ids processing"):
        tkz = ids_method(data[i])
        ids.append(tkz)
    ids = pad_sequences(
        ids, maxlen=MAX_LEN, dtype="int32", value=1, truncating="post", padding="post"
    )
    return ids


class PhoBertTraining:
    def __init__(self, ids_train, y_train, ids_test, y_test, classes) -> None:
        self.ids_train = ids_train
        self.ids_test = ids_test
        self.y_train = y_train
        self.y_test = y_test

        print("type of all dataset")
        print(type(ids_train), type(ids_test), type(y_train), type(y_test))

        self.classes = classes
        self.n_classes = len(self.classes)
        self.model = None

    def make_generator(self, vect_method, batch_size=512):
        print("==> Make data generator")
        self.train_generator = DataGenerator(
            self.ids_train,
            self.y_train,
            vect_method,
            batch_size=batch_size,
            n_classes=self.n_classes,
        )
        self.valid_generator = DataGenerator(
            self.ids_test,
            self.y_test,
            vect_method,
            batch_size=batch_size,
            n_classes=self.n_classes,
            shuffle=False,
        )

        print("==> Make data generator success")
        print("train_generator:", len(self.train_generator))
        print("valid_generator:", len(self.valid_generator))

    def build_model(self):
        model = Sequential()
        model.add(Conv1D(128, kernel_size=5, input_shape=(MAX_LEN, 768)))
        model.add(Conv1D(64, kernel_size=3, activation="relu"))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.n_classes, activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(model.summary())

        self.model = model

    def load_pretrain(self, path):
        self.model = tf.keras.models.load_model(path)

    def set_model(self, model):
        self.model = model

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

    def fit(self, n_epochs=10, log_dir="./logs", checkpoint_dir="data/models/"):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    checkpoint_dir, "model.{epoch:02d}-{val_loss:.2f}.h5"
                )
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        ]

        assert self.model is not None, "build model before fitting"
        assert (self.train_generator is not None) or (
            self.valid_generator is not None
        ), "making datagenerator before fitting"

        # fit the model
        history = self.model.fit(
            self.train_generator,
            epochs=n_epochs,
            verbose=1,
            validation_data=self.valid_generator,
            callbacks=callbacks,
        )
        # evaluate the model
        loss, accuracy = self.model.evaluate_generator(self.valid_generator, verbose=1)
        print("Accuracy: %f" % (accuracy * 100))
        print(f"Loss: {loss}")

        return history
