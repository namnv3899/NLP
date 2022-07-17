import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        ids,
        labels,
        vect_method,
        batch_size=16,
        max_seq_len=256,
        feature_len=768,
        n_classes=18,
        shuffle=True,
    ):
        self.max_seq_len = max_seq_len
        self.feature_len = feature_len
        self.batch_size = batch_size
        self.labels = labels
        self.ids = ids
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.vect_method = vect_method
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idx_temp):
        # X : (n_samples, *dim, n_channels)
        "Generates data containing batch_size samples"
        # Initialization
        X = np.empty((self.batch_size, self.max_seq_len, self.feature_len))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, idx in enumerate(idx_temp):
            X[
                i,
            ] = self.vect_method(self.ids[idx])
            # Store class
            y[i] = self.labels[idx]
        # X = X[:,:,:,np.newaxis]
        # y = to_categorical(y, num_classes=self.n_classes)
        # print(y.shape)
        return X, y
