"""
FastAPI Application Configuration

Implements production-grade configuration management with:
- Environment-based settings
- Secret management
- Type-safe configuration with Pydantic
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application
    APP_NAME: str = "Passive Bot Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = Field(
        default=["https://youruidai.gov.in"], env="ALLOWED_ORIGINS"
    )

    # MongoDB
    MONGODB_URL: str = Field(..., env="MONGODB_URL")
    MONGODB_DB_NAME: str = Field(default="bot_detection", env="MONGODB_DB_NAME")
    MONGODB_MAX_POOL_SIZE: int = 100
    MONGODB_MIN_POOL_SIZE: int = 10

    # Redis
    REDIS_URL: str = Field(..., env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: List[str] = Field(
        default=["localhost:9092"], env="KAFKA_BOOTSTRAP_SERVERS"
    )
    KAFKA_SIGNAL_TOPIC: str = "bot-detection-signals"
    KAFKA_VERDICT_TOPIC: str = "bot-detection-verdicts"
    KAFKA_TRAINING_TOPIC: str = "bot-detection-training"

    # ML Service
    ML_SERVICE_URL: str = Field(default="http://ml-service:8001", env="ML_SERVICE_URL")
    ML_SERVICE_TIMEOUT: int = 5  # seconds
    ML_BATCH_SIZE: int = 32
    ML_INFERENCE_TIMEOUT_MS: int = 100

    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    HMAC_KEY: str = Field(..., env="HMAC_KEY")
    SESSION_KEY_ROTATION_HOURS: int = 24
    REPLAY_ATTACK_WINDOW_SECONDS: int = 300  # 5 minutes

    # Rate Limiting
    RATE_LIMIT_PER_IP_PER_MINUTE: int = 1000
    RATE_LIMIT_PER_SESSION_PER_MINUTE: int = 100
    RATE_LIMIT_ENABLED: bool = True

    # Verification Thresholds
    BOT_PROBABILITY_ALLOW_THRESHOLD: float = 0.1  # < 0.1 = definitely human
    BOT_PROBABILITY_CHALLENGE_THRESHOLD: float = 0.7  # > 0.7 = likely bot
    CONFIDENCE_THRESHOLD: float = 0.5  # Minimum confidence for decision

    # Data Retention (hours)
    SIGNAL_TTL_HOURS: int = 24
    FEATURE_VECTOR_TTL_HOURS: int = 168  # 7 days
    VERDICT_TTL_HOURS: int = 720  # 30 days

    # Caching
    VERDICT_CACHE_TTL_SECONDS: int = 300  # 5 minutes
    FEATURE_CACHE_TTL_SECONDS: int = 60
    CACHE_ENABLED: bool = True

    # Monitoring
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = True
    JAEGER_AGENT_HOST: str = Field(default="localhost", env="JAEGER_AGENT_HOST")
    JAEGER_AGENT_PORT: int = Field(default=6831, env="JAEGER_AGENT_PORT")

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text
    LOG_FILE: Optional[str] = None

    # Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: int = 30
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: str = "Exception"

    # Feature Flags
    ENABLE_ONLINE_LEARNING: bool = True
    ENABLE_DRIFT_DETECTION: bool = True
    ENABLE_EXPLAINABILITY: bool = True
    ENABLE_ADAPTIVE_CHALLENGES: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_cors(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("KAFKA_BOOTSTRAP_SERVERS", pre=True)
    def parse_kafka_servers(cls, v):
        if isinstance(v, str):
            return [server.strip() for server in v.split(",")]
        return v

    @property
    def mongodb_settings(self):
        """Get MongoDB connection settings"""
        return {
            "url": self.MONGODB_URL,
            "db_name": self.MONGODB_DB_NAME,
            "maxPoolSize": self.MONGODB_MAX_POOL_SIZE,
            "minPoolSize": self.MONGODB_MIN_POOL_SIZE,
        }

    @property
    def redis_settings(self):
        """Get Redis connection settings"""
        return {
            "url": self.REDIS_URL,
            "max_connections": self.REDIS_MAX_CONNECTIONS,
            "socket_timeout": self.REDIS_SOCKET_TIMEOUT,
            "socket_connect_timeout": self.REDIS_SOCKET_CONNECT_TIMEOUT,
        }

    @property
    def kafka_settings(self):
        """Get Kafka connection settings"""
        return {
            "bootstrap_servers": self.KAFKA_BOOTSTRAP_SERVERS,
            "client_id": f"{self.APP_NAME}-{self.ENVIRONMENT}",
        }

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection helper"""
    return settings
