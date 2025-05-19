from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print('tensorflow version:', tf.__version__)

class MemoryDNN:
    def __init__(self, net, learning_rate=0.01, training_interval=10, batch_size=100, memory_size=1000, output_graph=False):
        # net is a list like [input_dim, ..., output_dim], where input_dim = N*3.
        self.net = net  
        self.training_interval = training_interval  
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.enumerate_actions = []
        self.memory_counter = 1
        self.cost_his = []
        # Memory shape: [memory_size, input_dim + output_dim]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        self._build_net()

    def _build_net(self):
        # Use Conv1D layers with padding 'same' to preserve dimensions.
        self.model = keras.Sequential([
            layers.Conv1D(32, 3, activation='relu', padding='same', input_shape=[int(self.net[0] / 3), 3]),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.net[-1], activation='sigmoid')
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=['accuracy'])
        self.model.summary()

    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))
        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        if self.memory_counter < self.batch_size:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=True)
        elif self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)
        batch_memory = self.memory[sample_index, :]
        input_dim = self.net[0]  # input_dim = N*3
        h_train = batch_memory[:, :input_dim].reshape(self.batch_size, int(self.net[0] / 3), 3)
        m_train = batch_memory[:, input_dim:]
        hist = self.model.fit(h_train, m_train, verbose=0)
        loss_val = hist.history['loss'][0]
        if loss_val <= 0:
            print("Warning: Training loss is zero or negative. Loss =", loss_val)
            loss_val = 1e-6  # set a small positive value
        self.cost_his.append(loss_val)

    def decode(self, h, k=1, mode='OP'):
        # h: numpy array of shape (N*3,)
        num_users = int(self.net[0] / 3)
        h = h.reshape(num_users, 3)
        h = h[np.newaxis, :]  # shape becomes (1, num_users, 3)
        m_pred = self.model.predict(h, verbose=0)
        # For now, simply threshold the prediction to produce a binary action.
        m_candidate = (m_pred[0] > 0.5).astype(int)
        return m_pred[0], [m_candidate]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.title('Conv-based MemoryDNN Training Cost')
        plt.show()
