import pymongo
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from app.utils.logging import logger

client: AsyncIOMotorClient = None
db = None
is_mongodb_connected = False

async def connect_to_mongo():
    global client, db, is_mongodb_connected
    try:
        logger.info(f"Attempting MongoDB connection to {settings.MONGODB_URI}...")
        client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)
        # Test connection ping
        await client.admin.command('ping')
        db = client[settings.MONGODB_DB_NAME]
        is_mongodb_connected = True
        logger.info("Successfully connected to MongoDB Database!")
    except Exception as e:
        is_mongodb_connected = False
        logger.warning(f"MongoDB connection failed: {e}. Active fallback mode: IN-MEMORY REPOSITORY.")

async def close_mongo_connection():
    global client, is_mongodb_connected
    if client:
        client.close()
        logger.info("Closed MongoDB connection.")
        is_mongodb_connected = False
