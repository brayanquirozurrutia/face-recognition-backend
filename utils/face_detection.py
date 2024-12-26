import mediapipe as mp
import cv2
import numpy as np

# Mediapipe config for face detection
mp_face_detection = mp.solutions.face_detection

def detect_faces(image: np.ndarray):
    """
    Detect faces in an image using Mediapipe
    :param image: Image to process
    :return: List of faces detected
    """
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return []
        faces = []
        for i, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            # Save detected faces to disk for validation
            cropped_face = image[y:y + h, x:x + w]
            cv2.imwrite(f"face_detected_{i}.jpg", cropped_face)
            faces.append((x, y, w, h))
        return faces
