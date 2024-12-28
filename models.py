from sqlalchemy import Column, Integer, String, LargeBinary
from hashlib import md5
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    face_id = Column(LargeBinary)  # Store the embedding as a binary blob
    face_id_hash = Column(String, unique=True, index=True)  # Hash of the embedding for faster search

    @staticmethod
    def generate_hash(embedding: bytes) -> str:
        """
        Generate an MD5 hash of the embedding for storage and search.
        :param embedding: The embedding to hash
        :return: MD5 hash of the embedding
        """
        return md5(embedding).hexdigest()

class DetectedFace(Base):
    __tablename__ = "detected_faces"
    id = Column(Integer, primary_key=True, index=True)
    face_id = Column(String, unique=True, index=True, nullable=False)
    image = Column(LargeBinary)  # Almacena la imagen en binario (como blob)