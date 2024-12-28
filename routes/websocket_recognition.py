from fastapi import WebSocket, WebSocketDisconnect
from database import get_db
from utils.face_detection import detect_faces, save_face_to_db
from utils.face_recognition import extract_embedding, compare_embeddings
from models import User
import cv2
import numpy as np

async def websocket_endpoint(websocket: WebSocket):
    """
    Process the WebSocket connection and receive the image frames
    :param websocket: WebSocket connection
    :return: None
    """

    # Accept the WebSocket connection
    await websocket.accept()

    # Get the database session
    db = next(get_db())
    try:
        while True:
            # Get the image bytes
            data = await websocket.receive_bytes()

            # Convert the bytes to an image
            np_frame = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

            # Check if the frame is valid
            if frame is None:
                print("Error decoding frame")
                continue

            # Detect faces and save to Redis and DB
            faces = await detect_faces(frame)

            # Save the faces to the database
            for face in faces:
                face_id = face["id"]
                await save_face_to_db(face_id)

            # Recognize registered users
            recognized = []
            for face in faces:
                x, y, w, h = face["bbox"]
                cropped_face = cv2.resize(frame[y:y + h, x:x + w], (160, 160))
                embedding = extract_embedding(cropped_face)

                for user in db.query(User).all():
                    user_embedding = np.frombuffer(user.face_id, dtype=np.float32)
                    if compare_embeddings(embedding, user_embedding):
                        recognized.append({"id": user.id, "name": user.name})
                        break

            # Send the recognized faces to the client
            await websocket.send_json({"recognized_faces": recognized})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        db.close()
