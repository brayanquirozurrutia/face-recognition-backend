# Face Recognition Backend

This project implements a backend application for facial recognition, including detecting faces, registering users, and recognizing users from uploaded images. Built using FastAPI, PostgreSQL, and Redis, this application showcases a basic implementation of a face recognition pipeline using Mediapipe and Facenet-PyTorch.

## Features

1. **Face Detection:** Detects faces in uploaded images using Mediapipe.
2. **User Registration:** Registers users with facial embeddings extracted from uploaded images.
3. **Face Recognition:** Recognizes registered users by comparing facial embeddings using cosine similarity.

## Architecture

- **FastAPI**: RESTful API framework for Python.
- **PostgreSQL**: Database to store user data and embeddings.
- **Redis**: Cache layer for future extensions (e.g., real-time data processing).
- **Facenet-PyTorch**: Model for extracting facial embeddings.
- **Mediapipe**: Library for detecting faces in images.

## Requirements

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/face-recognition-backend.git
cd face-recognition-backend
```

2. Start the services with Docker:

```bash
docker-compose up -d
```

3. Run the FastAPI application:

```bash
uvicorn main:app --reload
```

4. Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Endpoints

### Detect Faces
**POST** `/api/recognition/detect`

- **Description**: Detect faces in an uploaded image.
- **Input**: Image file (as `file` parameter in form-data).
- **Output**:
  ```json
  {
      "faces": [[x, y, w, h], ...]
  }
  ```

### Register User
**POST** `/api/recognition/register`

- **Description**: Register a user with their facial embedding.
- **Input**:
  - `name` (string): Name of the user.
  - `file` (image): Image of the user's face.
- **Output**:
  ```json
  {
      "message": "Usuario registrado con éxito",
      "id": 1
  }
  ```

### Recognize Face
**POST** `/api/recognition/recognize`

- **Description**: Recognize registered users in an uploaded image.
- **Input**: Image file (as `file` parameter in form-data).
- **Output**:
  ```json
  {
      "recognized_faces": [
          {"id": 1, "name": "Brayan Quiroz"}
      ]
  }
  ```

## Directory Structure

```
face-recognition-backend/
├── database.py           # Database connection and setup
├── docker-compose.yml    # Docker configuration
├── main.py               # FastAPI application setup
├── models.py             # SQLAlchemy models
├── redis_connection.py   # Redis connection setup
├── requirements.txt      # Python dependencies
├── routes/
│   └── recognition.py    # API endpoints
├── utils/
│   ├── face_detection.py # Mediapipe-based face detection
│   └── face_recognition.py # Embedding extraction and comparison
```

## Author

[Brayan Quiroz Urrutia](https://github.com/<your-username>)

