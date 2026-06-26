import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Bidirectional, GaussianNoise, Attention, GlobalAveragePooling1D
from tensorflow.keras.losses import CategoricalFocalCrossentropy

def build_hybrid_model(frames_per_sequence=30, landmarks_per_frame=268, vocab_size=250):
    
    raw_coord_input = Input(shape=(frames_per_sequence, landmarks_per_frame))
    noisy_coords = GaussianNoise(0.05)(raw_coord_input)
    
    spatial_features = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(noisy_coords)
    spatial_features = BatchNormalization()(spatial_features)
    spatial_features = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(spatial_features)
    spatial_features = BatchNormalization()(spatial_features)
    
    downsampled_features = MaxPooling1D(pool_size=2)(spatial_features)
    
    temporal_memory = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'))(downsampled_features)
    temporal_memory = Dropout(0.5)(temporal_memory)
    
    temporal_memory_deep = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'))(temporal_memory)
    temporal_memory_deep = Dropout(0.5)(temporal_memory_deep)

    apex_weights = Attention()([temporal_memory_deep, temporal_memory_deep])
    context_vector = GlobalAveragePooling1D()(apex_weights)

    dense_layer = Dense(256, activation='relu')(context_vector)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Dropout(0.5)(dense_layer)
    
    gloss_prediction = Dense(vocab_size, activation='softmax')(dense_layer)

    asl_model = Model(inputs=raw_coord_input, outputs=gloss_prediction)

    custom_loss = CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
    
    asl_model.compile(optimizer='Adam', loss=custom_loss, metrics=['categorical_accuracy'])
    
    return asl_model

if __name__ == "__main__":
    print("Initializing Golden Build Architecture...")
    model = build_hybrid_model()
    model.summary()