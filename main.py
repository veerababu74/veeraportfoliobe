from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.database import connect_db, close_db
from app.services.chatbot import init_chatbot
from app.routes import auth, profile, content, contact, upload, chatbot


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    init_chatbot()
    yield
    await close_db()


app = FastAPI(
    title="Portfolio API",
    description="Backend API for Veera's Portfolio Website",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
origins = [
    settings.FRONTEND_URL,
    "http://localhost:3000",
    "http://localhost:5173",
]
# Allow all Vercel preview URLs for this project
if settings.FRONTEND_URL and ".vercel.app" in settings.FRONTEND_URL:
    origins.append(settings.FRONTEND_URL.replace("https://", "https://*."))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(content.router)
app.include_router(contact.router)
app.include_router(upload.router)
app.include_router(chatbot.router)


@app.get("/")
async def root():
    return {"message": "Portfolio API is running!"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
