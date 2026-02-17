"""
Emotion Detection Module

Uses the 'deepface' library for robust face detection and emotion classification.
"""

import cv2
from deepface import DeepFace

class EmotionDetector:
    """
    Real-time emotion detection using the DeepFace library.
    """
    
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    EMOTION_COLORS = {
        'angry': (0, 0, 255),       # Red
        'disgust': (0, 165, 255),   # Orange
        'fear': (128, 0, 128),      # Purple
        'happy': (0, 255, 255),     # Yellow
        'sad': (255, 0, 0),         # Blue
        'surprise': (255, 105, 180),# Pink
        'neutral': (128, 128, 128)  # Gray
    }
    
    def __init__(self, model_path=None):
        """
        Initializes the emotion detector.
        DeepFace handles its own models, so model_path is ignored.
        """
        print("Initializing DeepFace model... This may take a moment on first run.")
        # Pre-load the model by analyzing a dummy frame to prevent a delay later.
        dummy_frame = (cv2.imread("blank.jpg") if cv2.imread("blank.jpg") is not None 
                       else cv2.UMat(224, 224, cv2.CV_8UC3))
        try:
            DeepFace.analyze(dummy_frame, actions=['emotion'], enforce_detection=False)
            print("DeepFace model initialized successfully.")
        except Exception as e:
            print(f"DeepFace initialized, but a minor startup error occurred: {e}")

    def detect(self, frame, confidence_threshold=0.5):
        """
        Detect emotion in a given frame using the DeepFace library.
        """
        annotated_frame = frame.copy()
        
        try:
            # Use enforce_detection=False to prevent crashes when no face is in the frame.
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                
                dominant_emotion = result['dominant_emotion']
                confidence = result['emotion'][dominant_emotion] / 100.0
                
                # --- THIS IS THE FIX ---
                # We access the dictionary keys directly instead of unpacking.
                region = result['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # Draw bounding box and label
                color = self.EMOTION_COLORS.get(dominant_emotion, (255, 255, 255))
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                
                label_text = f"{dominant_emotion.capitalize()} ({confidence:.2f})"
                cv2.putText(annotated_frame, label_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if confidence >= confidence_threshold:
                    return dominant_emotion.capitalize(), confidence, annotated_frame
                else:
                    # A face was detected, but confidence was too low. Show the box but don't return the emotion.
                    return None, confidence, annotated_frame
            else:
                # No face was detected by DeepFace.
                return None, 0.0, annotated_frame

        except Exception as e:
            # This will catch any other unexpected errors from DeepFace.
            # print(f"An error occurred during detection: {e}") # Uncomment for deep debugging
            return None, 0.0, annotated_frame

