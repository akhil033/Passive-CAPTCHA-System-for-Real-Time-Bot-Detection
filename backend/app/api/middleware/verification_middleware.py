"""
Verification Middleware

Core middleware that intercepts all protected requests and performs
passive bot detection. This is the primary enforcement point for the system.

Architecture:
1. Extract session and signature from request
2. Validate payload integrity (HMAC, timestamp, nonce)
3. Check Redis cache for recent verdict
4. Call ML inference service if no cached verdict
5. Make allow/challenge/block decision
6. Log verdict and publish to Kafka
7. Enforce decision

Performance targets:
- p95 latency: <50ms
- p99 latency: <100ms
- Cache hit rate: >80%
"""

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing import Optional, Dict, Any
import time
import hmac
import hashlib
import json
import structlog
from datetime import datetime, timedelta

from app.core.config import get_settings
from app.core.exceptions import (
    SignatureValidationError,
    ReplayAttackError,
    SessionNotFoundError,
    VerificationFailedError,
)
from app.infrastructure.cache.verdict_cache import VerdictCache
from app.infrastructure.ml.inference_client import MLInferenceClient
from app.infrastructure.messaging.kafka_producer import KafkaProducerClient
from app.domain.models.verdict import Verdict, VerdictDecision
from app.domain.services.signal_processor import SignalProcessor

logger = structlog.get_logger()
settings = get_settings()


class VerificationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for passive bot detection

    This middleware is applied to all protected routes and performs
    real-time verification based on behavioral signals.
    """

    def __init__(
        self,
        app,
        verdict_cache: VerdictCache,
        ml_client: MLInferenceClient,
        kafka_producer: KafkaProducerClient,
        signal_processor: SignalProcessor,
    ):
        super().__init__(app)
        self.verdict_cache = verdict_cache
        self.ml_client = ml_client
        self.kafka_producer = kafka_producer
        self.signal_processor = signal_processor

        # Paths that bypass verification (health checks, public assets)
        self.bypass_paths = {
            "/health",
            "/metrics",
            "/api/v1/session/create",
            "/docs",
            "/openapi.json",
            "/static",
        }

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "ml_inferences": 0,
            "decisions": {"ALLOW": 0, "CHALLENGE": 0, "BLOCK": 0},
        }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Main middleware entry point"""

        start_time = time.time()
        self.metrics["total_requests"] += 1

        # Check if path should bypass verification
        if self._should_bypass(request.path):
            return await call_next(request)

        # Extract verification payload from request
        try:
            verification_payload = await self._extract_verification_payload(request)
        except Exception as e:
            logger.error(
                "Failed to extract verification payload",
                error=str(e),
                path=request.url.path,
            )
            raise HTTPException(status_code=400, detail="Invalid verification payload")

        session_id = verification_payload.get("sessionId")

        # Inject session_id and trace_id into request state
        request.state.session_id = session_id
        request.state.trace_id = f"trace-{session_id}-{int(time.time() * 1000)}"

        log = logger.bind(
            session_id=session_id,
            trace_id=request.state.trace_id,
            path=request.url.path,
            method=request.method,
        )

        try:
            # Step 1: Validate signature and prevent replay attacks
            await self._validate_request_signature(verification_payload, log)

            # Step 2: Check cache for recent verdict
            cached_verdict = await self._get_cached_verdict(session_id, log)

            if cached_verdict:
                verdict = cached_verdict
                self.metrics["cache_hits"] += 1
                log.info("Verdict retrieved from cache", verdict=verdict.decision.value)
            else:
                # Step 3: Perform ML inference
                verdict = await self._perform_verification(verification_payload, log)
                self.metrics["ml_inferences"] += 1

                # Step 4: Cache the verdict
                await self._cache_verdict(session_id, verdict, log)

            # Step 5: Make decision
            decision = self._make_decision(verdict, log)
            self.metrics["decisions"][decision.value] += 1

            # Step 6: Publish verdict to Kafka for analytics
            await self._publish_verdict(verdict, log)

            # Step 7: Enforce decision
            if decision == VerdictDecision.BLOCK:
                log.warning("Request blocked", bot_probability=verdict.bot_probability)
                raise HTTPException(
                    status_code=403,
                    detail="Access denied. Suspicious activity detected.",
                )

            elif decision == VerdictDecision.CHALLENGE:
                log.info("Challenge required", bot_probability=verdict.bot_probability)
                # Inject challenge requirement into response
                request.state.challenge_required = True
                request.state.challenge_type = self._select_challenge_type(verdict)

            else:  # ALLOW
                log.info("Request allowed", bot_probability=verdict.bot_probability)
                request.state.challenge_required = False

            # Continue with request processing
            response = await call_next(request)

            # Add verification headers to response
            response.headers["X-Bot-Detection-Verdict"] = decision.value
            response.headers["X-Bot-Detection-Confidence"] = str(
                round(verdict.confidence, 2)
            )
            response.headers["X-Bot-Detection-Session"] = session_id

            # Log performance
            latency_ms = (time.time() - start_time) * 1000
            log.info(
                "Verification completed",
                latency_ms=round(latency_ms, 2),
                decision=decision.value,
                cached=cached_verdict is not None,
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            log.error("Verification failed with exception", error=str(e), exc_info=True)

            # Fail open in production (allow request but log for investigation)
            if settings.is_production:
                log.warning("Failing open due to error - allowing request")
                return await call_next(request)
            else:
                raise HTTPException(
                    status_code=500, detail=f"Verification error: {str(e)}"
                )

    def _should_bypass(self, path: str) -> bool:
        """Check if path should bypass verification"""
        return any(path.startswith(bypass_path) for bypass_path in self.bypass_paths)

    async def _extract_verification_payload(self, request: Request) -> Dict[str, Any]:
        """
        Extract verification payload from request headers

        Expected format:
        X-Bot-Detection-Payload: base64(json({
            sessionId, timestamp, nonce, signals, signature
        }))
        """
        payload_header = request.headers.get("X-Bot-Detection-Payload")

        if not payload_header:
            raise ValueError("Missing X-Bot-Detection-Payload header")

        # Decode base64 payload
        import base64

        try:
            payload_json = base64.b64decode(payload_header).decode("utf-8")
            payload = json.loads(payload_json)
        except Exception as e:
            raise ValueError(f"Invalid payload encoding: {str(e)}")

        # Validate required fields
        required_fields = ["sessionId", "timestamp", "nonce", "payload", "signature"]
        for field in required_fields:
            if field not in payload:
                raise ValueError(f"Missing required field: {field}")

        return payload

    async def _validate_request_signature(
        self, payload: Dict[str, Any], log: structlog.BoundLogger
    ) -> None:
        """
        Validate HMAC signature and prevent replay attacks

        Raises:
            SignatureValidationError: If signature is invalid
            ReplayAttackError: If timestamp is outside acceptable window or nonce reused
        """
        session_id = payload["sessionId"]
        timestamp = payload["timestamp"]
        nonce = payload["nonce"]
        signals = payload["payload"]
        provided_signature = payload["signature"]

        # Step 1: Validate timestamp (replay attack prevention)
        current_time_ms = int(time.time() * 1000)
        time_diff_ms = abs(current_time_ms - timestamp)

        if time_diff_ms > settings.REPLAY_ATTACK_WINDOW_SECONDS * 1000:
            log.warning(
                "Timestamp outside acceptable window",
                timestamp=timestamp,
                current_time=current_time_ms,
                diff_ms=time_diff_ms,
            )
            raise ReplayAttackError("Request timestamp outside acceptable window")

        # Step 2: Check nonce for replay attack
        nonce_key = f"nonce:{session_id}:{nonce}"
        nonce_exists = await self.verdict_cache.exists(nonce_key)

        if nonce_exists:
            log.warning("Nonce reuse detected - potential replay attack", nonce=nonce)
            raise ReplayAttackError("Nonce already used")

        # Store nonce with TTL
        await self.verdict_cache.set(
            nonce_key, "1", ttl_seconds=settings.REPLAY_ATTACK_WINDOW_SECONDS + 60
        )

        # Step 3: Validate HMAC signature
        # Recreate canonical string
        payload_hash = hashlib.sha256(signals.encode()).hexdigest()
        canonical_string = f"{session_id}|{timestamp}|{nonce}|{payload_hash}"

        # Compute expected signature
        expected_signature = hmac.new(
            settings.HMAC_KEY.encode(), canonical_string.encode(), hashlib.sha256
        ).hexdigest()

        # Constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(expected_signature, provided_signature):
            log.warning("Signature validation failed")
            raise SignatureValidationError("Invalid request signature")

        log.debug("Signature validated successfully")

    async def _get_cached_verdict(
        self, session_id: str, log: structlog.BoundLogger
    ) -> Optional[Verdict]:
        """Retrieve cached verdict from Redis"""

        if not settings.CACHE_ENABLED:
            return None

        cached_data = await self.verdict_cache.get(f"verdict:{session_id}")

        if cached_data:
            log.debug("Cache hit for verdict")
            return Verdict.parse_obj(cached_data)

        return None

    async def _perform_verification(
        self, payload: Dict[str, Any], log: structlog.BoundLogger
    ) -> Verdict:
        """
        Perform ML-based verification

        Steps:
        1. Extract and decode signals
        2. Process signals into features
        3. Call ML inference service
        4. Create verdict
        """
        session_id = payload["sessionId"]

        # Decode and decompress signals
        signals_data = self._decode_signals(payload["payload"])

        # Process signals into feature vector
        feature_vector = await self.signal_processor.process_signals(
            session_id=session_id, signals=signals_data
        )

        log.debug("Signals processed into features", feature_count=len(feature_vector))

        # Call ML inference service
        inference_result = await self.ml_client.predict(
            session_id=session_id, features=feature_vector
        )

        # Create verdict
        verdict = Verdict(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            bot_probability=inference_result["bot_probability"],
            confidence=inference_result["confidence"],
            model_version=inference_result["model_version"],
            feature_vector=feature_vector,
            decision=VerdictDecision.PENDING,  # Decision made in next step
            explanation=inference_result.get("explanation", {}),
        )

        return verdict

    def _decode_signals(self, encoded_signals: str) -> Dict[str, Any]:
        """Decode and decompress signal payload"""
        import base64
        import gzip

        try:
            # Base64 decode
            compressed_data = base64.b64decode(encoded_signals)

            # Try gzip decompression (fallback to direct JSON parse)
            try:
                decompressed_data = gzip.decompress(compressed_data)
                signals = json.loads(decompressed_data.decode("utf-8"))
            except:
                # Not compressed, parse directly
                signals = json.loads(compressed_data.decode("utf-8"))

            return signals
        except Exception as e:
            logger.error("Failed to decode signals", error=str(e))
            raise ValueError(f"Invalid signal encoding: {str(e)}")

    def _make_decision(
        self, verdict: Verdict, log: structlog.BoundLogger
    ) -> VerdictDecision:
        """
        Make allow/challenge/block decision based on verdict

        Decision logic:
        - bot_probability < 0.1: ALLOW (high confidence human)
        - bot_probability 0.1-0.7: ALLOW with monitoring
        - bot_probability 0.7-0.9: CHALLENGE (uncertain, lean bot)
        - bot_probability > 0.9: BLOCK (high confidence bot)

        Also considers confidence score
        """
        bot_prob = verdict.bot_probability
        confidence = verdict.confidence

        # Low confidence - be conservative (allow but monitor)
        if confidence < settings.CONFIDENCE_THRESHOLD:
            log.info("Low confidence - allowing with monitoring", confidence=confidence)
            return VerdictDecision.ALLOW

        # High confidence human
        if bot_prob < settings.BOT_PROBABILITY_ALLOW_THRESHOLD:
            return VerdictDecision.ALLOW

        # Uncertain but leaning human
        if bot_prob < 0.5:
            return VerdictDecision.ALLOW

        # Uncertain but leaning bot - adaptive challenge
        if bot_prob < settings.BOT_PROBABILITY_CHALLENGE_THRESHOLD:
            if settings.ENABLE_ADAPTIVE_CHALLENGES:
                return VerdictDecision.CHALLENGE
            else:
                return VerdictDecision.ALLOW  # Fail safe

        # High confidence bot
        return VerdictDecision.BLOCK

    def _select_challenge_type(self, verdict: Verdict) -> str:
        """
        Select appropriate challenge type based on suspicion level

        Returns challenge type identifier
        """
        bot_prob = verdict.bot_probability

        if bot_prob < 0.6:
            return "simple_confirmation"  # "Click continue"
        elif bot_prob < 0.7:
            return "timed_interaction"  # Wait 3 seconds
        else:
            return "behavioral_test"  # Move mouse to target

    async def _cache_verdict(
        self, session_id: str, verdict: Verdict, log: structlog.BoundLogger
    ) -> None:
        """Cache verdict in Redis"""

        if not settings.CACHE_ENABLED:
            return

        await self.verdict_cache.set(
            f"verdict:{session_id}",
            verdict.dict(),
            ttl_seconds=settings.VERDICT_CACHE_TTL_SECONDS,
        )

        log.debug("Verdict cached")

    async def _publish_verdict(
        self, verdict: Verdict, log: structlog.BoundLogger
    ) -> None:
        """Publish verdict to Kafka for analytics and online learning"""

        try:
            await self.kafka_producer.produce(
                topic=settings.KAFKA_VERDICT_TOPIC,
                key=verdict.session_id,
                value=verdict.dict(),
            )
            log.debug("Verdict published to Kafka")
        except Exception as e:
            # Non-critical - log but don't fail request
            log.error("Failed to publish verdict to Kafka", error=str(e))

    def get_metrics(self) -> Dict[str, Any]:
        """Get middleware performance metrics"""
        total = self.metrics["total_requests"]
        cache_hit_rate = self.metrics["cache_hits"] / total if total > 0 else 0

        return {**self.metrics, "cache_hit_rate": round(cache_hit_rate, 3)}
