"""
CRNN Model Architecture for Raga Classification
"""

import tensorflow as tf
from tensorflow import keras
from typing import List, Optional


class RagaCRNN:
    """
    CRNN (CNN + LSTM) architecture for raga classification
    """
    
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
    
    def build_model(self, cnn_filters=[32, 64], lstm_units=[64], dropout_rate=0.2):
        """Build a CRNN model"""
        model = keras.Sequential([
            keras.layers.Input(shape=self.input_shape),
            
            # CNN layers
            keras.layers.Conv2D(cnn_filters[0], (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(dropout_rate),
            
            keras.layers.Conv2D(cnn_filters[1], (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(dropout_rate),
            
            # Reshape for LSTM
            keras.layers.Reshape((-1, cnn_filters[1])),
            
            # LSTM layer
            keras.layers.LSTM(lstm_units[0], return_sequences=False),
            keras.layers.Dropout(dropout_rate),
            
            # Dense layers
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_k_categorical_accuracy')]
        )
        
        return model


def create_baseline_cnn(input_shape, num_classes, learning_rate=0.001):
    """
    Create a baseline CNN model for raga classification
    Simpler than CRNN, better for smaller datasets
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape, name='input_layer'),
        
        # Block 1
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Block 2
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Block 3
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        # Global pooling
        keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(num_classes, activation='softmax')
    ], name='RagaCNN_Baseline')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_k_categorical_accuracy')]
    )
    
    return model
