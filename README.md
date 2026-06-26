# Real-Time ASL-to-English Translation Pipeline

**Syracuse University | Artificial Neural Networks Capstone** **Author:** Aryan Nitin Kokabankar  

## 📌 Project Overview
This repository contains an edge-efficient, real-time translation pipeline designed to bridge the gap between raw American Sign Language (ASL) gestures and grammatically correct English. 

Unlike traditional computer vision models that rely on heavy raw video frames (CNNs) or massive Transformer networks that struggle with edge performance, this project utilizes a highly optimized **Attention-LSTM Hybrid architecture**. By extracting geometric landmarks and mathematically isolating the temporal "apex" of gestures, the system achieves **30 FPS real-time inference on standard CPU hardware** with a peak validation accuracy of 78%.

## 🏗️ System Architecture (3-Stage Pipeline)

1. **Spatial-Temporal Data Extraction (MediaPipe):**
   - Extracts 134 geometric hand landmarks per frame.
   - Calculates dynamic frame-to-frame velocities (totaling 268 features per sequence) to capture the fluid motion of ASL, bypassing the high latency of raw pixel processing.
2. **Core Classifier (Attention-LSTM Hybrid "Golden Build"):**
   - A Bidirectional LSTM processes a continuous rolling 30-frame sequence to track spatial trajectories.
   - A custom **Temporal Attention Mechanism** dynamically assigns the highest mathematical weights to the defining frames of the gesture, actively filtering out transitional noise (e.g., hands returning to rest).
3. **Natural Language Generation (Gemini 2.5 Flash):**
   - Converts the raw sequence of ASL signs ("Gloss") into natural, conversational English using a strictly constrained LLM prompt, acting purely as a grammatical refiner.

## 📂 Repository Structure
* `model_architecture.py`: Contains the Keras build for the Golden Build Attention-LSTM.
* `preprocess_data.py`: Handles Kaggle parquet extraction, NaN padding, and velocity calculations.
* `train_model.py`: The main training marathon script with callbacks (EarlyStopping, ReduceLROnPlateau).
* `realtime_inference.py`: The live OpenCV pipeline that stitches together MediaPipe, the Keras model, and the Gemini API.
* `utils.py`: Helper functions for hand normalization and fingertip distance calculations.
* `asl_hybrid_model.h5`: The pre-trained, finalized weights for the Golden Build.
* `vocab.txt`: The mapped dictionary of target ASL signs.
* `requirements.txt`: Python package dependencies.

## ⚙️ Installation

1. Clone this repository to your local machine.
2. Ensure you have Python 3.9+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt