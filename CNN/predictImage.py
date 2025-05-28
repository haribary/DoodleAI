import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


model = tf.keras.models.load_model("50class_model_1.h5")
categories = ["airplane","apple","bird","book","bridge","bus","car","cat","chair","clock",
            "computer","diamond","dog","ear","eyeglasses","fish","flower","guitar","harp","hot air balloon",
            "hourglass","house","key","leaf","lightning","moon","mug","octopus","pants","pencil",
            "pizza","rainbow","rifle","sailboat","scissors","shovel","skyscraper","snake","snowflake","strawberry",
            "sun","sword","television","toothbrush","tree","trumpet","t-shirt","umbrella","violin","wine glass"]

def predict(img):#imput numpy array that is in format and grayscale
    if img.shape != (28, 28):
        raise ValueError("Expected image of shape (28, 28), got " + str(img.shape))
    plt.imshow(img, cmap="gray")
    plt.axis("off") 
    plt.show()
    img = img / 255.0      
    img = img.reshape(1, 28, 28, 1) 

    predictions = model.predict(img)[0]
    top3index = predictions.argsort()[-3:][::-1]
    top3 = [(categories[i], predictions[i] * 100) for i in top3index]
    
    return top3

