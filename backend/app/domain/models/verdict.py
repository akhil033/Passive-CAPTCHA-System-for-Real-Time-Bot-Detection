"""
Domain Models for Bot Detection System

Implements core business entities following Domain-Driven Design principles.
These models are technology-agnostic and represent the core business logic.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


class VerdictDecision(str, Enum):
    """Possible verification decisions"""

    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    BLOCK = "BLOCK"
    PENDING = "PENDING"


class SessionStatus(str, Enum):
    """Session lifecycle states"""

    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    SUSPICIOUS = "SUSPICIOUS"


class Verdict(BaseModel):
    """
    Verification verdict for a session

    Represents the outcome of bot detection analysis.
    Stored in MongoDB with TTL for privacy compliance.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    timestamp: datetime
    bot_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    decision: VerdictDecision
    model_version: str
    feature_vector: Dict[str, float]
    explanation: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    inference_time_ms: Optional[float] = None
    cached: bool = False
    challenge_type: Optional[str] = None

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    @validator("bot_probability", "confidence")
    def validate_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probability must be between 0 and 1")
        return round(v, 4)  # Limit precision

    def to_mongo_document(self) -> Dict[str, Any]:
        """Convert to MongoDB document format"""
        doc = self.dict()
        doc["_id"] = doc.pop("id")
        doc["created_at"] = self.timestamp
        # Add TTL index field
        doc["expires_at"] = datetime.utcnow() + timedelta(days=30)
        return doc

    @classmethod
    def from_mongo_document(cls, doc: Dict[str, Any]) -> "Verdict":
        """Create from MongoDB document"""
        doc["id"] = str(doc.pop("_id"))
        doc.pop("expires_at", None)  # Remove TTL field
        return cls(**doc)


class Signal(BaseModel):
    """
    Raw behavioral signals from frontend

    Ephemeral data - stored for 24h only for aggregation and debugging.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    timestamp: datetime

    # Browser entropy (stable characteristics)
    browser: Dict[str, Any] = Field(default_factory=dict)

    # Dynamic behavioral signals
    pointer: Dict[str, Any] = Field(default_factory=dict)
    keyboard: Dict[str, Any] = Field(default_factory=dict)
    scroll: Dict[str, Any] = Field(default_factory=dict)
    timing: Dict[str, Any] = Field(default_factory=dict)
    network: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    page_url: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None  # Hashed for privacy

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_mongo_document(self) -> Dict[str, Any]:
        """Convert to MongoDB document with TTL"""
        doc = self.dict()
        doc["_id"] = doc.pop("id")
        doc["created_at"] = self.timestamp
        # Add TTL index field (24 hours)
        doc["expires_at"] = datetime.utcnow() + timedelta(hours=24)
        return doc

    @validator("ip_address")
    def hash_ip(cls, v):
        """Hash IP address for privacy"""
        if v:
            import hashlib

            return hashlib.sha256(v.encode()).hexdigest()
        return v


class Session(BaseModel):
    """
    User session for bot detection

    Tracks session lifecycle and aggregated behavior.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime

    status: SessionStatus = SessionStatus.ACTIVE

    # Session security
    hmac_key: str  # Ephemeral HMAC key for this session
    key_rotation_at: datetime

    # Aggregated metrics
    request_count: int = 0
    verdict_count: int = 0
    challenge_count: int = 0
    block_count: int = 0

    # Latest verdict
    latest_verdict: Optional[VerdictDecision] = None
    latest_bot_probability: Optional[float] = None

    # Device fingerprint (hashed)
    device_fingerprint: Optional[str] = None

    # Risk score (0-1, computed from verdicts)
    risk_score: float = 0.0

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.utcnow() > self.expires_at

    def is_active(self) -> bool:
        """Check if session is active"""
        return self.status == SessionStatus.ACTIVE and not self.is_expired()

    def should_rotate_key(self) -> bool:
        """Check if HMAC key should be rotated"""
        return datetime.utcnow() > self.key_rotation_at

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()

    def record_verdict(self, verdict: Verdict):
        """Record a verdict and update metrics"""
        self.verdict_count += 1
        self.latest_verdict = verdict.decision
        self.latest_bot_probability = verdict.bot_probability

        if verdict.decision == VerdictDecision.CHALLENGE:
            self.challenge_count += 1
        elif verdict.decision == VerdictDecision.BLOCK:
            self.block_count += 1

        # Update risk score (exponential moving average)
        alpha = 0.3
        self.risk_score = (
            alpha * verdict.bot_probability + (1 - alpha) * self.risk_score
        )

        # Mark as suspicious if too many blocks
        if self.block_count > 3:
            self.status = SessionStatus.SUSPICIOUS

    def to_mongo_document(self) -> Dict[str, Any]:
        """Convert to MongoDB document"""
        doc = self.dict()
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo_document(cls, doc: Dict[str, Any]) -> "Session":
        """Create from MongoDB document"""
        doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


class ModelMetadata(BaseModel):
    """
    ML model metadata for versioning and tracking

    Tracks model versions, performance metrics, and deployment status.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    version: str
    name: str
    description: Optional[str] = None

    # Model artifacts
    rf_model_path: Optional[str] = None
    xgb_model_path: Optional[str] = None
    feature_names: List[str] = Field(default_factory=list)

    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    false_positive_rate: Optional[float] = None

    # Deployment status
    status: str = "training"  # training, validation, staging, production, archived
    deployed_at: Optional[datetime] = None

    # Training metadata
    trained_at: datetime = Field(default_factory=datetime.utcnow)
    training_dataset_size: Optional[int] = None
    training_duration_seconds: Optional[float] = None

    # Feature importance
    feature_importance: Dict[str, float] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def is_production(self) -> bool:
        """Check if model is in production"""
        return self.status == "production"

    def to_mongo_document(self) -> Dict[str, Any]:
        """Convert to MongoDB document"""
        doc = self.dict()
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo_document(cls, doc: Dict[str, Any]) -> "ModelMetadata":
        """Create from MongoDB document"""
        doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


class FeatureVector(BaseModel):
    """
    Processed feature vector for ML inference

    Cached for reuse and drift detection.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Feature values (73 features)
    features: Dict[str, float]

    # Metadata
    feature_version: str = "1.0.0"
    processing_time_ms: Optional[float] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def validate_features(self, expected_features: List[str]) -> bool:
        """Validate that all expected features are present"""
        missing = set(expected_features) - set(self.features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return True

    def to_mongo_document(self) -> Dict[str, Any]:
        """Convert to MongoDB document with TTL"""
        doc = self.dict()
        doc["_id"] = doc.pop("id")
        # Add TTL index field (7 days)
        doc["expires_at"] = datetime.utcnow() + timedelta(days=7)
        return doc


class AuditLog(BaseModel):
    """
    Audit log entry for compliance and debugging

    Records all significant events in the system.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    event_type: str  # verification, session_create, model_update, etc.
    session_id: Optional[str] = None
    user_id: Optional[str] = None  # If authenticated

    # Event details
    decision: Optional[VerdictDecision] = None
    bot_probability: Optional[float] = None
    confidence: Optional[float] = None

    # Context
    ip_address_hash: Optional[str] = None
    user_agent_hash: Optional[str] = None
    request_path: Optional[str] = None

    # Model information
    model_version: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_mongo_document(self) -> Dict[str, Any]:
        """Convert to MongoDB document with TTL"""
        doc = self.dict()
        doc["_id"] = doc.pop("id")
        # Add TTL index field (90 days for audit logs)
        doc["expires_at"] = datetime.utcnow() + timedelta(days=90)
        return doc


# Export all models
__all__ = [
    "Verdict",
    "VerdictDecision",
    "Signal",
    "Session",
    "SessionStatus",
    "ModelMetadata",
    "FeatureVector",
    "AuditLog",
]
