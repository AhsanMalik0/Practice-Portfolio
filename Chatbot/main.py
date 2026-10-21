from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from app.config import PORT
from app.services.rag_service import rag
from app.api.routes import init_routes

@asynccontextmanager
async def lifespan(_):
    rag()  # Connect to MongoDB & index documents at startup
    yield

app = FastAPI(title="Store Chatbot API (Gemini)", lifespan=lifespan)
init_routes(app)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)