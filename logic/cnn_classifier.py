from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization


class Classifier(Model):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(2, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        return x


class Classifier2(Model):
    def __init__(self):
        super(Classifier2, self).__init__()

        self.conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.bn2 = BatchNormalization()
        self.mp2 = MaxPooling2D(pool_size=(2, 2))
        self.dp1 = Dropout(0.25)

        self.conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.bn3 = BatchNormalization()
        self.dp3 = Dropout(0.25)

        self.conv4 = Conv2D(128, kernel_size=(3, 3), activation='relu')
        self.bn4 = BatchNormalization()
        self.mp4 = MaxPooling2D(pool_size=(2, 2))
        self.dp4 = Dropout(0.25)

        self.flatten = Flatten()

        self.d5 = Dense(512, activation='relu')
        self.bn5 = BatchNormalization()
        self.dp5 = Dropout(0.5)

        self.d6 = Dense(128, activation='relu')
        self.bn6 = BatchNormalization()
        self.dp6 = Dropout(0.5)

        self.d7 = Dense(2, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.dp1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.dp3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.mp4(x)
        x = self.dp4(x)

        x = self.flatten(x)

        x = self.d5(x)
        x = self.bn5(x)
        x = self.dp5(x)

        x = self.d6(x)
        x = self.bn6(x)
        x = self.dp6(x)

        x = self.d7(x)

        return x

