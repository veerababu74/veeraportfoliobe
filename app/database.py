from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

client: AsyncIOMotorClient = None
db = None


async def connect_db():
    global client, db
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client.portfolio_db
    print("[OK] Connected to MongoDB")


async def close_db():
    global client
    if client:
        client.close()
        print("[CLOSED] MongoDB connection closed")


def get_db():
    return db
