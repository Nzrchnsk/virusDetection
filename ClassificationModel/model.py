from tensorflow.keras import models
from tensorflow.keras import layers


def main():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, 'relu'),
        layers.Dense(1, 'sigmoid')  # 1 neuron mal or util
    ])


if __name__ == '__main__':
    main()
