#RNN,VAE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def loadData():
    data= np.load("processed_data.npz")
    drawings = data['data']
    labels = data['labels']   
    scale = data['scale'] 
    masks = data['masks']

    print(f"Loaded {len(drawings)} drawings")
    print(f"Each drawing has {drawings.shape[1]} steps")
    print(f"Each step has {drawings.shape[2]} values [dx, dy, p1, p2, p3]")
    print(f"Coordinate scale factor: {scale}")
    print(labels.shape)
    print(masks.shape)
    
    return drawings,labels,masks,scale

drawings, labels, masks, scale = loadData()

input_size = 5
hidden_size = 256 #small enough to train efficiently, large enough to encode menaingful structure
latent_size = 128 #also what paper used
num_classes=50


def buildEncoder(input_size,hidden_size,latent_size,num_classes): #bidirectional lstm processes the strokes in both forward and reverse directions. 
    #gives both past and future context for each stroke step. Concatenate so effectively u get a 512 dimensional vector
    #returns a model that outputs mu and log(sigma) of a gaussian distributon of z(latent vector)
    sequence_input = layers.Input(shape=(None, input_size), name='sequence_input')
    mask_input = layers.Input(shape=(None,), dtype=tf.bool, name='mask_input') #boolean tensor indicates steps which are padding
    class_input = layers.Input(shape=(), dtype=tf.int32, name='class_input')

    class_embedding = layers.Embedding(input_dim=num_classes, output_dim=64, name='class_embedding'
    )(class_input)#embedding layer to onvert each class ID into learnable dense vector(better than one hot)

    sequence_length = tf.shape(sequence_input)[1]
    class_emb_expanded = tf.expand_dims(class_embedding, 1)  # (batch, 1, embedding_dim)
    class_emb_tiled = tf.tile(class_emb_expanded, [1, sequence_length, 1])  # (batch, seq_len, embedding_dim)
    enhanced_input = layers.Concatenate(axis=-1)([sequence_input, class_emb_tiled]) 
    #^ essentially makes the encoder "aware" of the class during every stroke step so it learns class conditional representations
    

    #defiine bidirectional lstm layer
    bi_lstm = layers.Bidirectional(
        layers.LSTM(hidden_size, return_sequences=False, return_state=True),
        merge_mode=None
    ) #outputs from forward and backward are returned separately

    lstm_outputs = bi_lstm(enhanced_input, mask=mask_input) #(fw output,fw hidden state,fw cellstate, bw output, bw h, bw c)
    #hidden states summarize sequence from both directions
    forward_h = lstm_outputs[1]  #Forward hidden state
    backward_h = lstm_outputs[4]  #Backward hidden state
    combined_hidden = layers.Concatenate()([forward_h, backward_h])#this is summary representation of whole input in one vector
    final_hidden = layers.Concatenate()([combined_hidden, class_embedding])#want to combine what model knows from sketch and the class

    #in latent space every drawing is a single fixed size vector like a cloud where more similar drawings are closer. Each drawing can
    #be mapped modeled by a gaussian which represents the uncertainty where that input lives in latent space.


    mu = layers.Dense(latent_size)(final_hidden)
    logSigma = layers.Dense(latent_size)(final_hidden)

    encoder_model = keras.Model(#define encdoer model
    inputs=[sequence_input, mask_input, class_input], 
    outputs=[mu, logSigma], 
    name='encoder'
    )

    return encoder_model

def buildDecoder(output_size,hidden_size,latent_size,num_classes):
    #Takes latent vector z and generates next drawing stroke
    #z is sampled from mu and logSigma in encoder
    input = layers.Input(shape=(latent_size,), name='latent_input')


