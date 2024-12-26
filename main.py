from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import Base, engine
from redis_connection import redis
from routes import recognition

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables in the database on startup
    Base.metadata.create_all(bind=engine)
    try:
        await redis.ping()
        print("Connected to Redis")
    except Exception as e:
        print(f"Redis connection failed: {e}")

    # Yield control back to FastAPI for the application to run
    yield

    # Close the database connection on shutdown and close the Redis connection
    await redis.close()

app = FastAPI(lifespan=lifespan)

app.include_router(recognition.router, prefix="/api/recognition", tags=["Recognition"])

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
