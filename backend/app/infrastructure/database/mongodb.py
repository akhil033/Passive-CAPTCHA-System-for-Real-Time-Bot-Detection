"""
MongoDB Database Schema Initialization

Creates collections, indexes, and TTL policies for production deployment.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING
from datetime import datetime, timedelta
import asyncio


class MongoDBSchemaInitializer:
    """Initialize MongoDB schema for bot detection system"""

    def __init__(self, mongodb_url: str, db_name: str):
        self.client = AsyncIOMotorClient(mongodb_url)
        self.db = self.client[db_name]

    async def initialize_all(self):
        """Initialize all collections and indexes"""
        print("🔧 Initializing MongoDB schema...")

        await self.init_sessions_collection()
        await self.init_signals_collection()
        await self.init_verdicts_collection()
        await self.init_feature_vectors_collection()
        await self.init_model_metadata_collection()
        await self.init_audit_logs_collection()

        print("✅ MongoDB schema initialized successfully")

    async def init_sessions_collection(self):
        """Initialize sessions collection"""
        collection = self.db.sessions

        indexes = [
            IndexModel([("session_id", ASCENDING)], unique=True),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0),  # TTL index
            IndexModel([("device_fingerprint", ASCENDING)]),
            IndexModel([("risk_score", DESCENDING)]),
        ]

        await collection.create_indexes(indexes)
        print("✓ Sessions collection initialized")

    async def init_signals_collection(self):
        """Initialize signals collection (24h TTL)"""
        collection = self.db.signals

        indexes = [
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0),  # 24h TTL
            IndexModel([("ip_address", ASCENDING)]),
        ]

        await collection.create_indexes(indexes)
        print("✓ Signals collection initialized (24h TTL)")

    async def init_verdicts_collection(self):
        """Initialize verdicts collection (30d TTL)"""
        collection = self.db.verdicts

        indexes = [
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("decision", ASCENDING)]),
            IndexModel([("bot_probability", DESCENDING)]),
            IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0),  # 30d TTL
            IndexModel([("model_version", ASCENDING)]),
        ]

        await collection.create_indexes(indexes)
        print("✓ Verdicts collection initialized (30d TTL)")

    async def init_feature_vectors_collection(self):
        """Initialize feature vectors collection (7d TTL)"""
        collection = self.db.feature_vectors

        indexes = [
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0),  # 7d TTL
            IndexModel([("feature_version", ASCENDING)]),
        ]

        await collection.create_indexes(indexes)
        print("✓ Feature vectors collection initialized (7d TTL)")

    async def init_model_metadata_collection(self):
        """Initialize model metadata collection (no TTL)"""
        collection = self.db.model_metadata

        indexes = [
            IndexModel([("version", ASCENDING)], unique=True),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("trained_at", DESCENDING)]),
            IndexModel([("deployed_at", DESCENDING)]),
        ]

        await collection.create_indexes(indexes)
        print("✓ Model metadata collection initialized")

    async def init_audit_logs_collection(self):
        """Initialize audit logs collection (90d TTL)"""
        collection = self.db.audit_logs

        indexes = [
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("event_type", ASCENDING)]),
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0),  # 90d TTL
            IndexModel([("decision", ASCENDING)]),
        ]

        await collection.create_indexes(indexes)
        print("✓ Audit logs collection initialized (90d TTL)")

    async def close(self):
        """Close MongoDB connection"""
        self.client.close()


async def main():
    """Main initialization function"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGODB_DB_NAME", "bot_detection")

    initializer = MongoDBSchemaInitializer(mongodb_url, db_name)

    try:
        await initializer.initialize_all()
    finally:
        await initializer.close()


if __name__ == "__main__":
    asyncio.run(main())
