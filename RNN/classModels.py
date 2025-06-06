import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split



def loadData():
    data = np.load("processed_data.npz")
    drawings = data['data']
    labels = data['labels']   
    scale = data['scale'] 
    masks = data['masks']

    print(f"Loaded {len(drawings)} drawings")
    print(f"Each drawing has {drawings.shape[1]} steps")
    print(f"Each step has {drawings.shape[2]} values [dx, dy, p1, p2, p3]")
    print(f"Coordinate scale factor: {scale}")
    print(f"Labels shape: {labels.shape}")
    print(f"Masks shape: {masks.shape}")
    
    return drawings, labels, masks, scale

# Load data
drawings, labels, masks, scale = loadData()

# Limit dataset size
# drawings = drawings[:10000]
# labels = labels[:10000]
# masks = masks[:10000]

def split_data_by_class(drawings, labels, masks):
    
    #Split data by class and return dictionaries indexed by class_id
    
    class_data = defaultdict(list)
    class_masks = defaultdict(list)
    
    for i, label in enumerate(labels):
        class_data[label].append(drawings[i])
        class_masks[label].append(masks[i])
    
    # Convert lists to numpy arrays
    for class_id in class_data:
        class_data[class_id] = np.array(class_data[class_id])
        class_masks[class_id] = np.array(class_masks[class_id])
        
        print(f"Class {class_id}: {len(class_data[class_id])} samples")
    
    return class_data, class_masks


def denormalize_coords(strokes, scale):
    strokes = strokes.copy()
    strokes[:, 0:2] *= scale
    return strokes

input_size = 5 
output_size = 5        
hidden_size = 256    

def buildDecoder(input_size, output_size, hidden_size, max_seq_len):
    # Build decoder-only model for a specific class.
    
    # Input layer - just the sequence
    sequence_input = layers.Input(shape=(max_seq_len-1, input_size), name='sequence_input')
    
    # LSTM layers for sequence generation
    x = layers.LSTM(hidden_size, return_sequences=True, name='lstm1')(sequence_input)
    x = layers.LSTM(hidden_size, return_sequences=True, name='lstm2')(x)
    
    # Output projections
    coords = layers.Dense(2, name='coordinates')(x)  # dx, dy
    pen_logits = layers.Dense(3, name='pen_states')(x)  # p1, p2, p3 (logits)
    
    # Combine outputs
    decoder_output = layers.Concatenate(axis=-1, name='final_output')([coords, pen_logits])
    
    decoder_model = keras.Model(
        inputs=sequence_input,
        outputs=decoder_output,
        name='class_specific_decoder'
    )
    
    return decoder_model

class classModel(keras.Model):
    #model wrapper for decoder with functionalities
    def __init__(self, input_size, output_size, hidden_size, max_seq_len, class_id, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.class_id = class_id
        
        # Build decoder for this specific class
        self.decoder = buildDecoder(
            input_size, output_size, hidden_size, max_seq_len
        )
        
        # Loss tracking
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.coord_loss_tracker = keras.metrics.Mean(name="coord_loss")
        self.pen_loss_tracker = keras.metrics.Mean(name="pen_loss")
    
    def get_config(self):
        
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'max_seq_len': self.max_seq_len,
            'class_id': self.class_id,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
       
        model_config = {k: v for k, v in config.items() 
                       if k in ['input_size', 'output_size', 'hidden_size', 'max_seq_len', 'class_id']}
        return cls(**model_config)
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.coord_loss_tracker,
            self.pen_loss_tracker,
        ]
    
    def call(self, inputs, training=None):
        #forward pass return next step
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:  # Training mode: (sequence_input, mask_input)
                sequence_input, _ = inputs
            else:
                sequence_input = inputs[0]
        else:
            sequence_input = inputs

        return self.decoder(sequence_input, training=training)
    
    def train_step(self, data):
        #custom train step for autoregressive training
        (sequence_input, mask_input), target_sequence = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self([sequence_input, mask_input], training=True)
            
            # Coordinate loss (MSE for dx, dy)
            coord_pred = predictions[:, :, :2]
            coord_target = target_sequence[:, :, :2]
            coord_loss = tf.reduce_mean(
                tf.square(coord_pred - coord_target) * 
                tf.expand_dims(tf.cast(mask_input, tf.float32), -1)
            )
            
            # Pen state loss (cross-entropy for categorical)
            pen_pred = predictions[:, :, 2:]
            pen_target = target_sequence[:, :, 2:]
            pen_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=pen_target,
                    logits=pen_pred
                ) * tf.cast(mask_input, tf.float32)
            )
            
            # Total loss
            total_loss = coord_loss + pen_loss
        
        # Update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.coord_loss_tracker.update_state(coord_loss)
        self.pen_loss_tracker.update_state(pen_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "coord_loss": self.coord_loss_tracker.result(),
            "pen_loss": self.pen_loss_tracker.result(),
        }
    
    def test_step(self, data):
        #validationstep to match train
        (sequence_input, mask_input), target_sequence = data
        
        # Forward pass without training
        predictions = self([sequence_input, mask_input], training=False)
        
        # Same loss calculation as train_step
        coord_pred = predictions[:, :, :2]
        coord_target = target_sequence[:, :, :2]
        coord_loss = tf.reduce_mean(
            tf.square(coord_pred - coord_target) * 
            tf.expand_dims(tf.cast(mask_input, tf.float32), -1)
        )
        
        pen_pred = predictions[:, :, 2:]
        pen_target = target_sequence[:, :, 2:]
        pen_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=pen_target,
                logits=pen_pred
            ) * tf.cast(mask_input, tf.float32)
        )
        
        total_loss = coord_loss + pen_loss
        
        # Update validation metrics
        self.total_loss_tracker.update_state(total_loss)
        self.coord_loss_tracker.update_state(coord_loss)
        self.pen_loss_tracker.update_state(pen_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "coord_loss": self.coord_loss_tracker.result(),
            "pen_loss": self.pen_loss_tracker.result(),
        }
    
def splitData(class_drawings, class_masks, test_size=0.2):
    #split data to train and val and also prepare target data
    if len(class_drawings) < 10:  # Skip classes with too few samples
        return None, None
    
    # Split data
    X_train, X_test, mask_train, mask_test = train_test_split(
        class_drawings, class_masks, test_size=test_size, random_state=42
    )
    
    # For autoregressive training:
    # Input: sequence[:-1]  
    # Target: sequence[1:]  (next step prediction)
    train_inputs = (X_train[:, :-1], mask_train[:, :-1])
    train_targets = X_train[:, 1:]
    
    test_inputs = (X_test[:, :-1], mask_test[:, :-1])
    test_targets = X_test[:, 1:]
    
    return (train_inputs, train_targets), (test_inputs, test_targets)

def train_class_model(model, train_data, test_data, epochs=50, batch_size=64):
    #train one class model
    train_inputs, train_targets = train_data
    test_inputs, test_targets = test_data
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_targets))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.7, 
            patience=5
        ),
        keras.callbacks.ModelCheckpoint(
            f'more_best_class_{model.class_id}.h5', 
            save_best_only=True, 
            monitor='val_loss'
        )
    ]
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Dummy loss - not actually used
    )
    
    # Train
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def trainAllModels(class_data, class_masks, input_size, output_size, hidden_size, max_seq_len, epochs=50):
    """
    Train separate models for each class
    """
    trained_models = {}
    training_histories = {}
    
    for class_id in class_data:
        print(f"\n=== Training model for Class {class_id} ===")
        
        # Prepare data for this class
        data_result = splitData(class_data[class_id], class_masks[class_id])
        
        train_data, test_data = data_result
        
        # Create model for this class
        model = classModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            class_id=class_id
        )
        model.summary()
        # Train the model
        history = train_class_model(model, train_data, test_data, epochs=epochs)
        
        # Store results
        trained_models[class_id] = model
        training_histories[class_id] = history
        
        print(f"Completed training for Class {class_id}")
    
    return trained_models, training_histories



def plotHistory(history, class_id):
    """Plot training and validation losses for a specific class"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Training History - Class {class_id}')
    
    # Total loss
    axes[0,0].plot(history.history['loss'], label='Train Loss')
    axes[0,0].plot(history.history['val_loss'], label='Val Loss')
    axes[0,0].set_title('Total Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Coordinate loss
    axes[0,1].plot(history.history['coord_loss'], label='Train Coord Loss')
    axes[0,1].plot(history.history['val_coord_loss'], label='Val Coord Loss')
    axes[0,1].set_title('Coordinate Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Pen loss
    axes[1,0].plot(history.history['pen_loss'], label='Train Pen Loss')
    axes[1,0].plot(history.history['val_pen_loss'], label='Val Pen Loss')
    axes[1,0].set_title('Pen State Loss')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Learning rate (if available)
    if 'learning_rate' in history.history:
        axes[1,1].plot(history.history['learning_rate'])
        axes[1,1].set_title('Learning Rate')
        axes[1,1].grid(True)
    else:
        axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

# class_data, class_masks = split_data_by_class(drawings, labels, masks)
# max_seq_len = 250  # Get from your data
# print('starting training')
# trained_models, histories = trainAllModels(
#     class_data, class_masks, input_size, output_size, 
#     hidden_size, max_seq_len, epochs=50)
# for class_id, history in histories.items():
#         print(f"Plotting history for Class {class_id}")
#         plotHistory(history, class_id)