import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models

import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet_1d(n_times, n_feature, n_out, units, drop=0.5):
    """
    Builds a 1D U-Net model for time series data with batch normalization and sigmoid activation for the output layer.
    Parameters:
    - n_times: int, the number of time steps in the input data (time series length).
    - n_feature: int, the number of features (or input bands) for each time step.
    - n_out: int, the number of output features (for regression) or output classes.
    - units: int, the number of filters in the first layer (other layers will scale based on this).
    - drop: float, the dropout rate (default is 0.5).
    
    Returns:
    - model: A Keras Model representing the 1D U-Net architecture.
    """
    
    inputs = layers.Input(shape=(n_times, n_feature))
    
    # Encoder (contracting path)
    c1 = layers.Conv1D(filters=units, kernel_size=3, activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)  # Batch normalization
    c1 = layers.Conv1D(filters=units, kernel_size=3, activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)  # Batch normalization
    p1 = layers.MaxPooling1D(pool_size=2)(c1)
    p1 = layers.Dropout(drop)(p1)
    
    c2 = layers.Conv1D(filters=units*2, kernel_size=3, activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)  # Batch normalization
    c2 = layers.Conv1D(filters=units*2, kernel_size=3, activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)  # Batch normalization
    p2 = layers.MaxPooling1D(pool_size=2)(c2)
    p2 = layers.Dropout(drop)(p2)
    
    c3 = layers.Conv1D(filters=units*4, kernel_size=3, activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)  # Batch normalization
    c3 = layers.Conv1D(filters=units*4, kernel_size=3, activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)  # Batch normalization
    p3 = layers.MaxPooling1D(pool_size=2)(c3)
    p3 = layers.Dropout(drop)(p3)
    
    c4 = layers.Conv1D(filters=units*8, kernel_size=3, activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)  # Batch normalization
    c4 = layers.Conv1D(filters=units*8, kernel_size=3, activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)  # Batch normalization
    p4 = layers.MaxPooling1D(pool_size=2)(c4)
    p4 = layers.Dropout(drop)(p4)
    
    # Bottleneck
    c5 = layers.Conv1D(filters=units*16, kernel_size=3, activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)  # Batch normalization
    c5 = layers.Conv1D(filters=units*16, kernel_size=3, activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)  # Batch normalization
    c5 = layers.Dropout(drop)(c5)
    
    # Decoder (expanding path)
    u6 = layers.Conv1DTranspose(filters=units*8, kernel_size=2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv1D(filters=units*8, kernel_size=3, activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)  # Batch normalization
    c6 = layers.Conv1D(filters=units*8, kernel_size=3, activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)  # Batch normalization
    c6 = layers.Dropout(drop)(c6)
    
    u7 = layers.Conv1DTranspose(filters=units*4, kernel_size=2, strides=2, padding='same')(c6)
    if u7.shape[1]<c3.shape[1]:
        # Padding configuration: dynamically retrieve first dimension size.
        padding = [[0, 0], [0, c3.shape[1]-u7.shape[1]], [0, 0]]  # Padding the second dimension by 1 at the end
        # Apply the padding
        u7 = tf.pad(u7, paddings=padding)
    
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv1D(filters=units*4, kernel_size=3, activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)  # Batch normalization
    c7 = layers.Conv1D(filters=units*4, kernel_size=3, activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)  # Batch normalization
    c7 = layers.Dropout(drop)(c7)
    
    u8 = layers.Conv1DTranspose(filters=units*2, kernel_size=2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv1D(filters=units*2, kernel_size=3, activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)  # Batch normalization
    c8 = layers.Conv1D(filters=units*2, kernel_size=3, activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)  # Batch normalization
    c8 = layers.Dropout(drop)(c8)
    
    u9 = layers.Conv1DTranspose(filters=units, kernel_size=2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv1D(filters=units, kernel_size=3, activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)  # Batch normalization
    c9 = layers.Conv1D(filters=units, kernel_size=3, activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)  # Batch normalization
    c9 = layers.Dropout(drop)(c9)
    
    # Output layer with sigmoid activation
    outputs = layers.Conv1D(filters=n_out, kernel_size=1, activation='sigmoid')(c9)
    
    # Model creation
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# # Simulate data
# n_times = 244  # Time series length
# n_feature = 1  # Number of input features (bands) at each time point
# n_out = 1  # Number of output features (for regression)

# # Generate random input data and output data
# X_train = np.random.rand(1000, n_times, n_feature).astype(np.float32)
# y_train = np.random.rand(1000, n_times, n_out).astype(np.float32)

# # Build the model with specified parameters
# model = build_unet_1d(n_times=n_times, n_feature=n_feature, n_out=n_out, units=64, drop=0.1)

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model on the simulated data
# history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# # Output the history to inspect the training process
# print(history.history)