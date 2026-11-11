from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

_client: AsyncIOMotorClient = None


async def connect_to_mongo():
    global _client
    _client = AsyncIOMotorClient(settings.mongodb_url)
    print(f"Connected to MongoDB at {settings.mongodb_url}")


async def close_mongo_connection():
    global _client
    if _client:
        _client.close()
        print("MongoDB connection closed")


def get_database():
    return _client[settings.mongodb_db_name]
