import numpy as np
import matplotlib.pyplot as plt
from predictImage import predict

data = np.load("data/bus.npy")
# print(data.shape)


image = data[1001].reshape(28, 28)
print("This image is: " + predict(image)[0][0])
    
plt.imshow(image, cmap="gray")
plt.axis("off") 
plt.show()
        



