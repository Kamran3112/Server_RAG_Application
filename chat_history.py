from pymongo import MongoClient
from datetime import datetime
from config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db["chat_history"]

def save_chat(file_id, session_id, messages):
    collection.update_one(
        {"file_id": file_id, "session_id": session_id},
        {"$set": {"messages": messages, "timestamp": datetime.utcnow()}},
        upsert=True
    )

def get_chat(file_id, session_id):
    chat = collection.find_one({"file_id": file_id, "session_id": session_id})
    return chat["messages"] if chat else []

def delete_chat(file_id: str, session_id: str = None):
    query = {"file_id": file_id}
    if session_id:
        query["session_id"] = session_id
    result = collection.delete_many(query)
    return result.deleted_count
