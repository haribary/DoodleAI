import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf


classNames=["airplane","apple","bird","book","bridge","bus","car","cat","chair","clock",
            "computer","diamond","dog","ear","eyeglasses","fish","flower","guitar","harp","hot air balloon",
            "hourglass","house","key","leaf","lightning","moon","mug","octopus","pants","pencil",
            "pizza","rainbow","rifle","sailboat","scissors","shovel","skyscraper","snake","snowflake","strawberry",
            "sun","sword","television","toothbrush","tree","trumpet","t-shirt","umbrella","violin","wine glass"] #50 classes

data=[]
labels=[]

for i in range (len(classNames)):
    data.append(np.load(f"CNN/data/{classNames[i]}.npy")[:3000])#first 1000 ea class
    labels.append(np.full((3000,), i))#1000 of each label



X = np.concatenate(data, axis=0)
y = np.concatenate(labels, axis=0)
X, y = shuffle(X, y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)

#normalizing
X_train = X_train / 255.0
X_val = X_val / 255.0

np.set_printoptions(linewidth=np.inf)  # Never wrap
print(X_train[0][:,:,0])
print("Class name:", classNames[y_train[0]])
model = tf.keras.Sequential([
    #1st conv layer
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    #2nd
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
    #output layers
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50, activation='softmax')
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15,validation_data=(X_val,y_val))


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

model.save("50class_model_2.h5")
