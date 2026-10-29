import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from app.database import mongodb
from app.utils.logging import logger

# Add project root to sys.path for security module
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from security.biometric_crypto import encrypt_template, decrypt_template

# In-memory stores fallback
_in_memory_users: Dict[str, Dict[str, Any]] = {}
_in_memory_logs: List[Dict[str, Any]] = []
_in_memory_face_templates: Dict[str, Dict[str, Any]] = {}


class UserRepository:
    @staticmethod
    async def create_user(user_data: Dict[str, Any]) -> str:
        user_id = str(uuid.uuid4())
        user_data["user_id"] = user_id
        user_data["created_at"] = datetime.now(timezone.utc).isoformat()
        
        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                await mongodb.db.users.insert_one(user_data)
                return user_id
            except Exception as e:
                logger.error(f"Failed to insert user to MongoDB: {e}. Falling back to in-memory.")

        _in_memory_users[user_data["email"].lower()] = dict(user_data)
        _in_memory_users[user_id] = dict(user_data)
        return user_id

    @staticmethod
    async def get_by_email(email: str) -> Optional[Dict[str, Any]]:
        email = email.lower()
        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                user = await mongodb.db.users.find_one({"email": email})
                if user:
                    user["_id"] = str(user["_id"])
                    return user
            except Exception as e:
                logger.error(f"Failed to query MongoDB by email: {e}")

        return _in_memory_users.get(email)

    @staticmethod
    async def get_by_id(user_id: str) -> Optional[Dict[str, Any]]:
        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                user = await mongodb.db.users.find_one({"user_id": user_id})
                if user:
                    user["_id"] = str(user["_id"])
                    return user
            except Exception as e:
                logger.error(f"Failed to query MongoDB by user_id: {e}")

        return _in_memory_users.get(user_id)


class FaceTemplateRepository:
    """
    Repository for user face identity templates.
    Enforces AES-256-GCM AEAD encrypted storage (zero plaintext embedding storage).
    """
    @staticmethod
    async def save_template(user_id: str, template_data: Dict[str, Any]) -> str:
        template_id = str(uuid.uuid4())
        
        # Extract plaintext embedding if present and perform AEAD encryption
        raw_embedding = template_data.pop("embedding", None)
        if raw_embedding is not None:
            encrypted_payload = encrypt_template(raw_embedding)
            template_data.update(encrypted_payload)

        template_data["template_id"] = template_id
        template_data["user_id"] = user_id
        template_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        if "created_at" not in template_data:
            template_data["created_at"] = template_data["updated_at"]

        # Ensure plaintext embedding key is completely removed
        template_data.pop("embedding", None)

        record_copy = dict(template_data)
        _in_memory_face_templates[user_id] = record_copy

        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                await mongodb.db.face_templates.update_one(
                    {"user_id": user_id},
                    {"$set": record_copy},
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Failed to save face template to MongoDB: {e}. Falling back to in-memory.")

        return template_id

    @staticmethod
    async def get_template_by_user_id(user_id: str) -> Optional[Dict[str, Any]]:
        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                doc = await mongodb.db.face_templates.find_one({"user_id": user_id})
                if doc:
                    doc["_id"] = str(doc["_id"])
                    return doc
            except Exception as e:
                logger.error(f"Failed to query face template by user_id: {e}")

        return _in_memory_face_templates.get(user_id)

    @staticmethod
    async def delete_template(user_id: str) -> bool:
        deleted = False
        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                res = await mongodb.db.face_templates.delete_one({"user_id": user_id})
                if res.deleted_count > 0:
                    deleted = True
            except Exception as e:
                logger.error(f"Failed to delete face template from MongoDB: {e}")

        if user_id in _in_memory_face_templates:
            del _in_memory_face_templates[user_id]
            deleted = True

        return deleted


_in_memory_voice_templates: Dict[str, Dict[str, Any]] = {}


class VoiceTemplateRepository:
    """
    Repository for user voice identity templates.
    Enforces AES-256-GCM AEAD encrypted storage (zero plaintext embedding storage).
    """
    @staticmethod
    async def save_template(user_id: str, template_data: Dict[str, Any]) -> str:
        template_id = str(uuid.uuid4())
        
        # Extract plaintext embedding if present and perform AEAD encryption
        raw_embedding = template_data.pop("embedding", None)
        if raw_embedding is not None:
            encrypted_payload = encrypt_template(raw_embedding)
            template_data.update(encrypted_payload)

        template_data["template_id"] = template_id
        template_data["user_id"] = user_id
        template_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        if "created_at" not in template_data:
            template_data["created_at"] = template_data["updated_at"]

        # Ensure plaintext embedding key is completely removed
        template_data.pop("embedding", None)

        record_copy = dict(template_data)
        _in_memory_voice_templates[user_id] = record_copy

        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                await mongodb.db.voice_templates.update_one(
                    {"user_id": user_id},
                    {"$set": record_copy},
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Failed to save voice template to MongoDB: {e}. Falling back to in-memory.")

        return template_id

    @staticmethod
    async def get_template_by_user_id(user_id: str) -> Optional[Dict[str, Any]]:
        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                doc = await mongodb.db.voice_templates.find_one({"user_id": user_id})
                if doc:
                    doc["_id"] = str(doc["_id"])
                    return doc
            except Exception as e:
                logger.error(f"Failed to query voice template by user_id: {e}")

        return _in_memory_voice_templates.get(user_id)

    @staticmethod
    async def delete_template(user_id: str) -> bool:
        deleted = False
        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                res = await mongodb.db.voice_templates.delete_one({"user_id": user_id})
                if res.deleted_count > 0:
                    deleted = True
            except Exception as e:
                logger.error(f"Failed to delete voice template from MongoDB: {e}")

        if user_id in _in_memory_voice_templates:
            del _in_memory_voice_templates[user_id]
            deleted = True

        return deleted


class AuthLogRepository:
    @staticmethod
    async def log_attempt(log_data: Dict[str, Any]) -> str:
        log_id = str(uuid.uuid4())
        log_data["id"] = log_id
        if "timestamp" not in log_data:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                await mongodb.db.auth_logs.insert_one(log_data)
                return log_id
            except Exception as e:
                logger.error(f"Failed to log auth attempt to MongoDB: {e}")

        _in_memory_logs.append(dict(log_data))
        return log_id

    @staticmethod
    async def get_all_logs(limit: int = 100, decision_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                query = {}
                if decision_filter and decision_filter.upper() != "ALL":
                    query["decision"] = decision_filter.upper()
                
                cursor = mongodb.db.auth_logs.find(query).sort("timestamp", -1).limit(limit)
                logs = []
                async for doc in cursor:
                    doc["_id"] = str(doc["_id"])
                    logs.append(doc)
                return logs
            except Exception as e:
                logger.error(f"Failed to fetch auth logs from MongoDB: {e}")

        # Fallback in-memory
        logs = list(reversed(_in_memory_logs))
        if decision_filter and decision_filter.upper() != "ALL":
            filter_val = decision_filter.upper()
            logs = [l for l in logs if l.get("decision") == filter_val or l.get("attack_type") == filter_val]
        return logs[:limit]

    @staticmethod
    async def get_log_by_id(log_id: str) -> Optional[Dict[str, Any]]:
        if mongodb.is_mongodb_connected and mongodb.db is not None:
            try:
                doc = await mongodb.db.auth_logs.find_one({"id": log_id})
                if doc:
                    doc["_id"] = str(doc["_id"])
                    return doc
            except Exception as e:
                logger.error(f"Failed to fetch log by ID from MongoDB: {e}")

        for l in _in_memory_logs:
            if l.get("id") == log_id:
                return l
        return None

    @staticmethod
    async def get_dashboard_stats() -> Dict[str, Any]:
        logs = await AuthLogRepository.get_all_logs(limit=500)
        total = len(logs)
        allowed = sum(1 for l in logs if l.get("decision") == "ALLOW")
        blocked = sum(1 for l in logs if l.get("decision") == "BLOCK")
        step_up = sum(1 for l in logs if l.get("decision") == "STEP-UP")
        deepfake_attacks = sum(1 for l in logs if l.get("deepfake_probability", 0) > 0.5 or l.get("attack_type") == "DEEPFAKE_ATTACK")
        
        avg_risk = round(sum(l.get("risk_score", 0) for l in logs) / max(1, total), 1)

        return {
            "total_attempts": total,
            "allowed_count": allowed,
            "blocked_count": blocked,
            "step_up_count": step_up,
            "deepfake_attacks_count": deepfake_attacks,
            "average_risk_score": avg_risk
        }
