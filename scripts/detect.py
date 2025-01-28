# scripts/detect.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def detect_deepfakes(video_path, model_path, target_size=(256, 256)):
    """
    Detect deepfakes in a video using the trained model.
    """
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, target_size)
        normalized_frame = resized_frame / 255.0
        input_data = np.expand_dims(normalized_frame, axis=0)

        prediction = model.predict(input_data)
        label = "Fake" if prediction[0] > 0.5 else "Real"

        # Display result
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Deepfake Detection', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    VIDEO_PATH = 'test_video.mp4'
    MODEL_PATH = 'results/trained_model.h5'

    detect_deepfakes(VIDEO_PATH, MODEL_PATH)

