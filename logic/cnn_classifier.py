from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D


class Classifier(Model):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)  # default activation is none

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        return x
