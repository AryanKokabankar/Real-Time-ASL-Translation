import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from preprocess_data import process_parquet_to_numpy
from model_architecture import build_hybrid_model

def load_and_prepare_data(csv_path, base_dir, target_signs, num_samples_per_sign=400):
    print("Loading dataset mapping...")
    df = pd.read_csv(csv_path)
    df = df[df['sign'].isin(target_signs)]
    
    X, y = [], []
    label_map = {label:num for num, label in enumerate(target_signs)}
    
    for sign in target_signs:
        print(f"Processing data for: {sign}...")
        sign_data = df[df['sign'] == sign]
        
        if len(sign_data) > 0:
            balanced_data = sign_data.sample(n=num_samples_per_sign, replace=True, random_state=42)
            
            for index, row in balanced_data.iterrows():
                file_path = os.path.join(base_dir, row['path'])
                if os.path.exists(file_path):
                    sequence = process_parquet_to_numpy(file_path)
                    if sequence is not None:
                        X.append(sequence)
                        y.append(label_map[sign])

    X = np.array(X, dtype=np.float32)
    y = to_categorical(y, num_classes=len(target_signs)) 
    return X, y, label_map

if __name__ == "__main__":
    CSV_FILE = "train.csv" 
    DATA_DIRECTORY = "kaggle_data" 
    
    print("Analyzing full dataset to map the entire dictionary...")
    df_full = pd.read_csv(CSV_FILE)
    
    my_target_signs = df_full['sign'].unique().tolist()
    print(f"\nTarget Vocabulary ({len(my_target_signs)} words mapped!)")
    
    with open('vocab.txt', 'w') as f:
        for word in my_target_signs:
            f.write(f"{word}\n")
            
    print("\nStarting Perfectly Balanced Data Pipeline... (Loading 100,000 sequences!)")
    
    X, y, labels = load_and_prepare_data(CSV_FILE, DATA_DIRECTORY, my_target_signs, num_samples_per_sign=400)
    
    print(f"\nData loaded! X shape: {X.shape}, y shape: {y.shape}")
    
    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = build_hybrid_model(num_classes=len(my_target_signs))
        
        checkpoint = ModelCheckpoint('asl_autosave.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1)
        early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=15, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

        print("\nStarting The 100K Balanced Marathon...")
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                            callbacks=[checkpoint, early_stopping, reduce_lr])
        
        model.save('asl_hybrid_model.h5')
        print("\nUltimate Model saved as 'asl_hybrid_model.h5'!")
    else:
        print("No data was loaded. Check your file paths.")