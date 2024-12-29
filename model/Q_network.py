import tensorflow as tf
from tensorflow.keras import layers

class Q_Network(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super(Q_Network, self).__init__(**kwargs)  # Pasa los argumentos adicionales a la clase base
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = layers.LSTM(hidden_size, return_sequences=True, dropout=0.3)
        self.fc = layers.Dense(output_size)

    def call(self, x):
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
        x = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

    def get_config(self):
        config = super(Q_Network, self).get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
