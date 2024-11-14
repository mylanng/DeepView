import tensorflow as tf
from tensorflow.keras import layers, Model

def unet_3d(input_shape=(128, 128, 128, 1)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)
    
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)
    
    # Bottleneck
    b1 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(p2)
    b1 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(b1)
    
    # Decoder
    u2 = layers.UpSampling3D((2, 2, 2))(b1)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c5)
    
    u1 = layers.UpSampling3D((2, 2, 2))(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u1)
    c6 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c6)
    
    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(c6)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
