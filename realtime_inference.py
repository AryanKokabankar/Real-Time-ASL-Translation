import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import google.generativeai as genai

import mediapipe.python.solutions as mp_solutions
mp_hands = mp_solutions.hands
mp_drawing = mp_solutions.drawing_utils

from utils import normalize_hand, get_fingertip_distances

genai.configure(api_key="AIzaSyAVsaDpZ3u71nT_cxxFFvK0ATlRAZXP8ik")
gemini_refiner = genai.GenerativeModel('gemini-2.5-flash')

def translate_gloss_to_english(gloss_buffer):
    if not gloss_buffer:
        return "No words detected."
        
    raw_gloss_string = " ".join(gloss_buffer)
    
    system_prompt = f"""
    You are an expert American Sign Language (ASL) translator. 
    I will provide you with a sequence of raw ASL words (gloss). 
    Your job is to translate this sequence into a natural, grammatically correct English sentence. 
    Do not add extra conversational filler, just translate the intent.
    
    Raw ASL: {raw_gloss_string}
    Natural English Translation:
    """
    
    try:
        api_response = gemini_refiner.generate_content(system_prompt)
        return api_response.text.strip()
    except Exception as e:
        return f"LLM Error: {e}"

print("Loading Golden Build Attention-LSTM...")
gloss_classifier = load_model('asl_hybrid_model.h5', compile=False) 

target_vocabulary = []
if os.path.exists('vocab.txt'):
    with open('vocab.txt', 'r') as f:
        target_vocabulary = [line.strip() for line in f.readlines()]
    print(f"Vocabulary mapped: {len(target_vocabulary)} ASL signs active.")
else:
    print("FATAL: vocab.txt not found! Pipeline cannot map categorical outputs.")
    exit()

def extract_live_keypoints(mediapipe_results):
    frame_keypoints = np.zeros(134) 
    
    if mediapipe_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(mediapipe_results.multi_hand_landmarks):
            handedness = mediapipe_results.multi_handedness[hand_idx].classification[0].label
            
            raw_coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])
            normalized_coords = normalize_hand(raw_coords)
            fingertip_dists = get_fingertip_distances(normalized_coords)
            
            hand_features = np.concatenate([normalized_coords, fingertip_dists])
            
            if handedness == 'Left':
                frame_keypoints[0:67] = hand_features
            elif handedness == 'Right':
                frame_keypoints[67:134] = hand_features
                
    return frame_keypoints

temporal_buffer = []
gloss_sentence_buffer = []
confidence_predictions = []
CONFIDENCE_THRESHOLD = 0.75 
final_english_output = "Waiting for translation... (Press 'T')"

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, cv_frame = cap.read()
        if not ret: break
            
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        mp_results = hands.process(rgb_frame)
        
        rgb_frame.flags.writeable = True
        bgr_render_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(bgr_render_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        extracted_frame_features = extract_live_keypoints(mp_results)
        temporal_buffer.append(extracted_frame_features)
        
        temporal_buffer = temporal_buffer[-30:]
        
        if len(temporal_buffer) == 30:
            spatial_tensor = np.array(temporal_buffer) 
            
            velocity_tensor = np.zeros_like(spatial_tensor)
            velocity_tensor[1:] = spatial_tensor[1:] - spatial_tensor[:-1] 
            
            inference_input = np.concatenate([spatial_tensor, velocity_tensor], axis=1)
            prediction_array = gloss_classifier.predict(np.expand_dims(inference_input, axis=0), verbose=0)[0]
            
            if mp_results.multi_hand_landmarks:
                current_top_guess = np.argmax(prediction_array)
                confidence_predictions.append(current_top_guess)
                
                if len(confidence_predictions) > 10:
                    confidence_predictions = confidence_predictions[-10:] 
                    stable_guess_idx = max(set(confidence_predictions), key=confidence_predictions.count)
                    
                    if prediction_array[stable_guess_idx] > CONFIDENCE_THRESHOLD:
                        detected_gloss = target_vocabulary[stable_guess_idx]
                        
                        if len(gloss_sentence_buffer) > 0:
                            if detected_gloss != gloss_sentence_buffer[-1]:
                                gloss_sentence_buffer.append(detected_gloss)
                        else:
                            gloss_sentence_buffer.append(detected_gloss)

        if len(gloss_sentence_buffer) > 6:
            gloss_sentence_buffer = gloss_sentence_buffer[-6:]

        cv2.rectangle(bgr_render_frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(bgr_render_frame, f"ASL: {' '.join(gloss_sentence_buffer)}", (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(bgr_render_frame, (0, 430), (640, 480), (16, 117, 245), -1)
        cv2.putText(bgr_render_frame, f"LLM: {final_english_output}", (10, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            if len(gloss_sentence_buffer) > 0:
                cv2.putText(bgr_render_frame, "TRANSLATING...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('ASL to English Edge Pipeline', bgr_render_frame)
                cv2.waitKey(1) 
                
                final_english_output = translate_gloss_to_english(gloss_sentence_buffer)
                
                gloss_sentence_buffer = [] 
                confidence_predictions = []

        cv2.imshow('ASL to English Edge Pipeline', bgr_render_frame)
            
    cap.release()
    cv2.destroyAllWindows()