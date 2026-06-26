import numpy as np

def normalize_hand(hand_landmarks):
    wrist = hand_landmarks[0]
    centered = hand_landmarks - wrist
    max_dist = np.max(np.abs(centered))
    if max_dist > 0:
        return (centered / max_dist).flatten()
    return centered.flatten()

def get_fingertip_distances(flat_normalized_hand):
    if np.all(flat_normalized_hand == 0):
        return np.zeros(4) 
        
    coords = flat_normalized_hand.reshape((21, 3))
    thumb_tip = coords[4]
    
    distances = []
    for tip in [8, 12, 16, 20]:
        dist = np.linalg.norm(thumb_tip - coords[tip])
        distances.append(dist)
        
    return np.array(distances)