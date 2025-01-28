# scripts/preprocess.py
import os
import cv2
from tqdm import tqdm
from face_recognition import face_locations

def extract_faces_from_video(video_path, output_dir, target_size=256):
    """
    Extracts faces from a video and saves them as individual images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the frame
        face_locations_list = face_locations(frame, model='cnn')
        for idx, (top, right, bottom, left) in enumerate(face_locations_list):
            face = frame[top:bottom, left:right]
            face_resized = cv2.resize(face, (target_size, target_size))
            output_path = os.path.join(output_dir, f"{frame_count}_{idx}.jpg")
            cv2.imwrite(output_path, face_resized)
        
        frame_count += 1

    cap.release()

def preprocess_dataset(raw_data_dir, processed_data_dir):
    """
    Preprocesses all videos in `raw_data_dir` and saves extracted faces to `processed_data_dir`.
    """
    categories = ['real', 'fake']
    for category in categories:
        input_dir = os.path.join(raw_data_dir, category)
        output_dir = os.path.join(processed_data_dir, category)

        videos = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi'))]
        for video in tqdm(videos, desc=f"Processing {category} videos"):
            video_path = os.path.join(input_dir, video)
            extract_faces_from_video(video_path, output_dir)

if __name__ == '__main__':
    RAW_DATA_DIR = 'data/'
    PROCESSED_DATA_DIR = 'data/processed/'

    preprocess_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
