import mediapipe as mp
import cv2
import numpy as np
import uuid
from redis_connection import redis
from database import get_db
from models import DetectedFace

# Mediapipe config for face detection
mp_face_detection = mp.solutions.face_detection

async def detect_faces(image: np.ndarray):
    """
    Detect faces in an image using Mediapipe Face Detection
    :param image: Image to process
    :return: List of faces detected
    """
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return []

        faces = []

        # Loop through the detected faces
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            # Cut the face and save it
            cropped_face = image[y:y + h, x:x + w]
            _, buffer = cv2.imencode(".jpg", cropped_face)
            face_id = str(uuid.uuid4())

            # Save to Redis (TTL of 60 seconds)
            await redis.setex(f"detected_face:{face_id}", 60, buffer.tobytes())

            # Append the face to the list
            faces.append({"id": face_id, "bbox": (x, y, w, h)})

        return faces


async def save_face_to_db(face_id):
    """
    Save a detected face to the database using the face_id
    :param face_id: Face ID to save
    :return: True if the face was saved successfully, False otherwise
    """

    # Get the face data from Redis
    face_data = await redis.get(f"detected_face:{face_id}")

    # Verify that the data was found
    if not face_data:
        print(f"No data found for face_id: {face_id}")
        return False

    # Verify that the data is binary
    if not isinstance(face_data, bytes):
        print(f"Unexpected data type for face_id {face_id}: {type(face_data)}")
        return False

    # Continue with saving to the database
    db = next(get_db())

    try:
        # Check if the face already exists in the database
        existing_face = db.query(DetectedFace).filter_by(face_id=face_id).first()
        if existing_face:
            print(f"Face with face_id {face_id} already exists in database.")
            return False

        # Save the face to the database
        new_face = DetectedFace(face_id=face_id, image=face_data)
        db.add(new_face)
        db.commit()
        return True
    except Exception as e:
        print(f"Error saving face to database: {e}")
        return False
    finally:
        db.close()