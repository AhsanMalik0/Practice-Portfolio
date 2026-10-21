import sys
from pymongo import MongoClient
from app.config import MONGODB_URI, MONGODB_DB, logging

log = logging.getLogger("bot")

def get_mongo_client():
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client[MONGODB_DB]
    except Exception as e:
        sys.exit(f"Cannot connect to MongoDB at {MONGODB_URI}: {e}")