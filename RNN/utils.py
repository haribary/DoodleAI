import numpy as np

# Load the .npz file
data = np.load("processed_data.npz")

# Access the arrays
X = data['data']      # shape: (samples, 250, 5)
y = data['labels']    # shape: (samples,)

# View shape info
print("Data shape:", X.shape)
print("Labels shape:", y.shape)

# View one sample (e.g., the first drawing)
print("Sample label:", y[0])
print("Sample strokes:\n", X[0])  # shape: (250, 5)

# Optional: view only the first few strokes (to reduce output clutter)
print("First 10 strokes:\n", X[0][:10])