from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from utils.face_detection import detect_faces
from utils.face_recognition import extract_embedding, compare_embeddings
from database import get_db
from models import User
import cv2
import numpy as np
from fastapi import HTTPException

router = APIRouter()


@router.post("/detect")
async def detect_faces_endpoint(file: UploadFile = File(...)):
    """
    Detect faces in an image
    :param file: Image file
    :return: List of faces detected
    """
    image = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    faces = detect_faces(image)
    return {"faces": faces}


@router.post("/recognize")
async def recognize_face(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Recognize faces in an image
    :param file: Image file
    :param db: Database session
    :return: List of recognized faces
    """
    image = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    faces = detect_faces(image)
    if not faces:
        return {"message": "No faces detected"}

    recognized = []
    for face in faces:
        x, y, w, h = face
        cropped_face = cv2.resize(image[y:y + h, x:x + w], (160, 160))
        embedding = extract_embedding(cropped_face)
        users = db.query(User).all()
        for user in users:
            user_embedding = np.frombuffer(user.face_id, dtype=np.float32)
            if compare_embeddings(embedding, user_embedding):
                recognized.append({"id": user.id, "name": user.name})
                break
    return {"recognized_faces": recognized}


@router.post("/register")
async def register_user(name: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Register a new user in the database
    :param name: Username
    :param file: Image file
    :param db: Database session
    :return: Message and user ID
    """
    image = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    faces = detect_faces(image)

    if not faces:
        raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

    # Take the first detected face
    x, y, w, h = faces[0]
    cropped_face = cv2.resize(image[y:y + h, x:x + w], (160, 160))
    embedding = extract_embedding(cropped_face)

    # Convert the embedding to bytes
    embedding_bytes = embedding.tobytes()

    # Generate the embedding hash
    embedding_hash = User.generate_hash(embedding_bytes)

    # Verify if the hash already exists in the database
    existing_user = db.query(User).filter(User.face_id_hash == embedding_hash).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="El usuario ya está registrado")

    # Create a new user
    new_user = User(name=name, face_id=embedding_bytes, face_id_hash=embedding_hash)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "Usuario registrado con éxito", "id": new_user.id}