import pandas as pd
import numpy as np
import os
from utils import normalize_hand, get_fingertip_distances

def process_parquet_to_numpy(file_path, target_frames=30):
    try:
        landmark_df = pd.read_parquet(file_path)
        active_frames = landmark_df['frame'].unique()
        
        processed_sequence = []
        
        for frame_idx in active_frames:
            current_frame = landmark_df[landmark_df['frame'] == frame_idx]
            
            raw_left = current_frame[current_frame['type'] == 'left_hand'][['x', 'y', 'z']].values
            raw_right = current_frame[current_frame['type'] == 'right_hand'][['x', 'y', 'z']].values
            
            if len(raw_left) == 0 or np.isnan(raw_left).all():
                left_coords = np.zeros(63) 
                left_finger_dist = np.zeros(4)
            else:
                raw_left = np.nan_to_num(raw_left) 
                left_coords = normalize_hand(raw_left)
                left_finger_dist = get_fingertip_distances(left_coords)
                
            if len(raw_right) == 0 or np.isnan(raw_right).all():
                right_coords = np.zeros(63)
                right_finger_dist = np.zeros(4)
            else:
                raw_right = np.nan_to_num(raw_right)
                right_coords = normalize_hand(raw_right)
                right_finger_dist = get_fingertip_distances(right_coords)
                
            combined_spatial = np.concatenate([left_coords, right_coords, left_finger_dist, right_finger_dist])
            processed_sequence.append(combined_spatial)
            
        processed_sequence = np.array(processed_sequence)
        
        if len(processed_sequence) < target_frames:
            padding = np.zeros((target_frames - len(processed_sequence), 134))
            processed_sequence = np.concatenate([processed_sequence, padding])
        else:
            processed_sequence = processed_sequence[:target_frames]
            
        velocities = np.zeros_like(processed_sequence)
        velocities[1:] = processed_sequence[1:] - processed_sequence[:-1]
        
        final_tensor = np.concatenate([processed_sequence, velocities], axis=1)
            
        return final_tensor
        
    except Exception as e:
        return None