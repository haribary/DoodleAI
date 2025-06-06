import tensorflow as tf
from tensorflow import keras
import numpy as np
from classModels import(
    denormalize_coords,
    classModel
)
import matplotlib.pyplot as plt

input_size = 5          # [dx, dy, p1, p2, p3]
output_size = 5         # Same as input
hidden_size = 256       # From your new code
max_seq_len = 250
scale = 10.276792526245117


models = {}

def loadAllModels():
    """Load all available class models at startup"""
    print("Loading all available models...")
    
    loaded_count = 0
    for class_id in range(50):  # Assuming you have classes 0-49
        model_file = f"more_best_class_{class_id}.h5"
        
        try:
            # Create model architecture
            model = classModel(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                max_seq_len=max_seq_len,
                class_id=class_id
            )
            
            # Load weights
            model.load_weights(model_file)
            models[class_id] = model
            loaded_count += 1
            
            print(f"Loaded model for class {class_id}")
            
        except Exception as e:
            print(f"Failed to load {model_file}: {e}")
    
    print(f"Successfully loaded {loaded_count} models")
    return loaded_count == 50

def generate_drawing(class_id, temperature, max_len):
    print(f"Generating drawing for class {class_id} temperature {temperature}")
    
    # Get the pre-loaded model for this class
    model = models[class_id]
    
    # Initialize sequence
    current_sequence = np.zeros((1, max_seq_len-1, 5))
    current_sequence[0, 0] = [0, 0, 1, 0, 0]  # Start with pen down
    
    generated_steps = []
    
    for step in range(min(max_len, max_seq_len-1)):
        # Get prediction from class-specific model
        prediction = model(current_sequence, training=False)
        next_step_pred = prediction[0, step]
        
        # Sample next step
        next_step = sample_next_step(next_step_pred.numpy(), temperature)
        generated_steps.append(next_step.copy())
        
        # Update sequence
        if step + 1 < max_seq_len - 1:
            current_sequence[0, step + 1] = next_step
        
        # Check for end
        pen_state = np.argmax(next_step[2:])
        if pen_state == 2:  # End of sequence
            break
    
    # Convert and denormalize
    generated_seq = np.array(generated_steps, dtype=np.float32)
    generated_seq = denormalize_coords(generated_seq, scale)
    
    return generated_seq

def sample_next_step(prediction, temperature=0.8):
    """Sample next step from model prediction with temperature"""
    # Extract coordinates and pen logits
    coords = prediction[:2]
    pen_logits = prediction[2:]
    
    # Apply temperature to coordinates (add noise)
    coords_sampled = coords + np.random.normal(0, temperature * 0.1, size=2)
    
    # Apply temperature to pen states (categorical sampling)
    pen_probs = tf.nn.softmax(pen_logits / temperature).numpy()
    pen_state = np.random.choice(3, p=pen_probs)
    pen_one_hot = np.zeros(3)
    pen_one_hot[pen_state] = 1.0
    
    return np.concatenate([coords_sampled, pen_one_hot])

def visualize_drawing(drawing_sequence, title="Generated Drawing", figsize=(8, 6)):
    """Visualize a drawing sequence as connected strokes"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert relative coordinates to absolute positions
    x, y = 0, 0
    x_coords, y_coords = [x], [y]
    pen_states = []
    
    for step in drawing_sequence:
        dx, dy = step[0], step[1]
        pen_state = np.argmax(step[2:])  # 0: pen down, 1: pen up, 2: end
        
        x += dx
        y += dy
        x_coords.append(x)
        y_coords.append(y)
        pen_states.append(pen_state)
        
        if pen_state == 2:  # End of sequence
            break
    
    # Plot strokes
    stroke_x, stroke_y = [], []
    
    for i, (x, y, pen) in enumerate(zip(x_coords[1:], y_coords[1:], pen_states)):
        if pen == 0:  # Pen down - draw line
            stroke_x.append(x)
            stroke_y.append(y)
        else:  # Pen up or end - finish current stroke
            if stroke_x:
                ax.plot(stroke_x, stroke_y, 'b-', linewidth=2)
                stroke_x, stroke_y = [], []
    
    # Finish last stroke
    if stroke_x:
        ax.plot(stroke_x, stroke_y, 'b-', linewidth=2)
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Flip Y axis to match drawing convention
    plt.tight_layout()
    plt.show()


# if loadAllModels():
#     print('loaded')



classNames=["airplane","apple","bird","book","bridge","bus","car","cat","chair","clock",
            "computer","diamond","dog","ear","eyeglasses","fish","flower","guitar","harp","hot air balloon",
            "hourglass","house","key","leaf","lightning","moon","mug","octopus","pants","pencil",
            "pizza","rainbow","rifle","sailboat","scissors","shovel","skyscraper","snake","snowflake","strawberry",
            "sun","sword","television","toothbrush","tree","trumpet","t-shirt","umbrella","violin","wine glass"] #50 classes


        

classid = int(input("Class ID: "))
temperature = float(input("Temperature: "))

model_file = f"more_best_class_{classid}.h5"
try:
    # Create model architecture
    model = classModel(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        max_seq_len=max_seq_len,
        class_id=classid
    )
    
    # Load weights
    model.load_weights(model_file)
    models[classid] = model
    
    print(f"Loaded model for class {classid}")
    
except Exception as e:
    print(f"Failed to load {model_file}: {e}")


count = 0
drawings = []
for i in range(5):
    drawings.append(generate_drawing(classid, temperature, max_seq_len))
    
for drawing in drawings:
    visualize_drawing(drawing, f"Class {classNames[classid]} - Temperature {temperature} - Steps {len(drawing)}")


