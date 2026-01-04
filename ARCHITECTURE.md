# Passive Bot Detection System - Production Architecture

## Executive Summary

This system is designed to protect UIDAI-scale government platforms from bot attacks and DDoS threats while completely removing traditional CAPTCHA friction. MVP results achieving ~97% accuracy, this production-grade implementation handles millions of requests per minute with sub-100ms latency overhead.

## System Overview

### Core Principles
- **Zero Friction for Humans**: Passive detection eliminates explicit challenges
- **Privacy First**: No PII collection, short-lived session data, GDPR/UIDAI compliant
- **Defense in Depth**: Multi-layered protection from API gateway to ML inference
- **Adaptive Intelligence**: Online learning continuously adapts to new attack patterns
- **Fail-Safe Design**: Graceful degradation under attack, never block legitimate users
- **Observable and Auditable**: Complete audit trail with ML explainability

## Architecture Layers

### 1. Edge Layer (API Gateway + CDN)
```
┌─────────────────────────────────────────────────────────┐
│  Cloudflare / AWS CloudFront / Azure Front Door         │
│  - Geographic load balancing                            │
│  - DDoS protection (L3/L4)                              │
│  - TLS termination                                      │
│  - Rate limiting (coarse-grained)                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Kong / Traefik API Gateway                             │
│  - Request signing verification                         │
│  - IP reputation checking                               │
│  - Circuit breaker patterns                             │
│  - Request routing and load balancing                   │
└─────────────────────────────────────────────────────────┘
```

### 2. Application Layer (FastAPI Services)

```
┌──────────────────────────────────────────────────────────┐
│              Verification Middleware                     │
│  ┌──────────────────────────────────────────────────┐    │
│  │  1. Extract session token and request signature  │    │
│  │  2. Validate payload integrity                   │    │
│  │  3. Check Redis for cached verdict               │    │ 
│  │  4. Call ML Inference Service                    │    │
│  │  5. Enforce decision (allow/challenge/block)     │    │
│  │  6. Log decision with audit trail                │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Protected Business APIs                                │
│  - User authentication                                  │
│  - Document services (Aadhaar operations)               │
│  - Biometric services                                   │
│  - Update services                                      │
└─────────────────────────────────────────────────────────┘
```

### 3. ML Inference Layer

```
┌─────────────────────────────────────────────────────────┐
│  Inference Service (FastAPI)                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Feature Extraction                              │   │
│  │  - Normalize signal data                         │   │
│  │  - Compute derived features                      │   │
│  │  - Historical context enrichment                 │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Ensemble Prediction                             │   │
│  │  - Random Forest (base model)                    │   │
│  │  - XGBoost (boosted model)                       │   │
│  │  - Weighted averaging with confidence            │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Decision Logic                                  │   │
│  │  - High confidence (>0.9) → Allow                │   │
│  │  - Medium (0.5-0.9) → Adaptive challenge         │   │
│  │  - Low (<0.5) → Block                            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 4. Data Layer

```
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│  MongoDB Cluster   │  │   Redis Cluster    │  │  Kafka Cluster     │
│  - Sessions        │  │   - Rate limiting  │  │  - Signal stream   │
│  - Signals (TTL)   │  │   - Verdict cache  │  │  - Verdict stream  │
│  - Verdicts        │  │   - Session state  │  │  - Training data   │
│  - Model metadata  │  │   - Feature cache  │  │  - Drift events    │
└────────────────────┘  └────────────────────┘  └────────────────────┘
```

### 5. ML Pipeline Layer

```
┌─────────────────────────────────────────────────────────┐
│  Training Pipeline                                      │
│  1. Data ingestion from Kaggle datasets + production    │
│  2. Feature engineering and transformation              │
│  3. Ensemble model training (RF + XGBoost)              │
│  4. Model validation and performance metrics            │
│  5. SHAP explainability analysis                        │
│  6. Model versioning and artifact storage               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Online Learning Pipeline (Kafka Streams)               │
│  1. Consume signal and verdict streams                  │
│  2. Detect distribution drift                           │
│  3. Aggregate feedback for retraining                   │
│  4. Trigger incremental model updates                   │
│  5. A/B test new models before rollout                  │
└─────────────────────────────────────────────────────────┘
```

## Signal Collection Strategy

### Non-PII Environmental Signals (73 Features)

#### 1. Browser Entropy (15 features)
- User-Agent fingerprint hash
- Screen resolution and color depth
- Timezone offset
- Language preferences
- Plugin enumeration hash
- Canvas fingerprint
- WebGL fingerprint
- Audio context fingerprint
- Font enumeration hash
- Hardware concurrency
- Device memory
- Platform and OS version
- Browser feature support bitmap
- Cookie enabled status
- Do Not Track status

#### 2. Pointer Dynamics (18 features)
- Movement trajectory entropy
- Velocity distribution (mean, std, percentiles)
- Acceleration patterns
- Jitter and smoothness metrics
- Click timing distribution
- Double-click intervals
- Hover duration statistics
- Bezier curve fitting error
- Direction change frequency
- Movement pause count
- Edge collision patterns
- Pressure sensitivity (if available)
- Tilt angles (if available)
- Pointer type (mouse/touch/pen)
- Movement predictability score
- Human-like irregularity index
- Click target accuracy
- Movement to click delay

#### 3. Keyboard Dynamics (12 features)
- Typing speed distribution
- Key press duration patterns
- Inter-key latency distribution
- Typing rhythm entropy
- Error rate and correction patterns
- Shift key usage patterns
- Modifier key combinations
- Key repeat behavior
- Flight time statistics
- Dwell time statistics
- Typing burst patterns
- Pause duration distribution

#### 4. Interaction Timing (10 features)
- Focus event timing
- Blur event timing
- Page visibility changes
- Session duration
- Idle time distribution
- Action frequency distribution
- Task completion time
- Form interaction duration
- Navigation timing
- Time-to-interactive metrics

#### 5. Scroll Behavior (8 features)
- Scroll velocity distribution
- Acceleration patterns
- Smoothness metrics
- Direction changes
- Momentum behavior
- Scroll depth patterns
- Fractional pixel movements
- Touch vs wheel scroll detection

#### 6. Network Characteristics (10 features)
- Request timing jitter
- Round-trip time statistics
- Connection type detection
- Downlink speed estimation
- Request ordering anomalies
- Concurrent connection patterns
- DNS resolution timing
- TLS handshake timing
- Keep-alive behavior
- Request retry patterns

### Signal Collection Flow

```
┌─────────────────────────────────────────────────────────┐
│  Flutter Web Application                                │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  JavaScript Interop Bridge                       │   │
│  │  - Capture browser-level events                  │   │
│  │  - Compute entropy features                      │   │
│  │  - Buffer signals locally (no immediate send)    │   │
│  └──────────────────────────────────────────────────┘   │
│                        ↓                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Signal Aggregator (Dart)                        │   │
│  │  - Aggregate signals over 5-10 second windows    │   │
│  │  - Compute statistical features                  │   │
│  │  - Apply privacy filters (remove PII)            │   │
│  └──────────────────────────────────────────────────┘   │
│                        ↓                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Payload Signing                                 │   │
│  │  - Sign with session-specific ephemeral key      │   │
│  │  - Add timestamp and nonce (replay prevention)   │   │
│  │  - Compress and encode                           │   │
│  └──────────────────────────────────────────────────┘   │
│                        ↓                                │
│            Send to Backend (on API calls)               │
└─────────────────────────────────────────────────────────┘
```

## Ensemble ML Model Architecture

### Model Design Rationale

**Why Random Forest + XGBoost?**
- **Random Forest**: Robust to outliers, handles feature interactions, provides baseline predictions
- **XGBoost**: Sequential error correction, better gradient optimization, handles imbalanced data
- **Ensemble**: Combines stability of RF with precision of XGBoost
- **Proven**: Achieved 97% accuracy in MVP with SIH evaluation

### Training Pipeline

```python
# Pseudo-code for ensemble strategy
class EnsembleVerificationModel:
    """
    Weighted ensemble of Random Forest and XGBoost
    Uses confidence-based weighting
    """
    
    def predict(self, features):
        # Get predictions from both models
        rf_proba = self.rf_model.predict_proba(features)
        xgb_proba = self.xgb_model.predict_proba(features)
        
        # Weighted average based on model confidence
        rf_confidence = self._compute_confidence(rf_proba)
        xgb_confidence = self._compute_confidence(xgb_proba)
        
        # Dynamic weighting
        rf_weight = rf_confidence / (rf_confidence + xgb_confidence)
        xgb_weight = 1 - rf_weight
        
        final_proba = rf_weight * rf_proba + xgb_weight * xgb_proba
        
        return {
            'bot_probability': final_proba[0][1],
            'confidence': (rf_confidence + xgb_confidence) / 2,
            'rf_vote': rf_proba[0][1],
            'xgb_vote': xgb_proba[0][1]
        }
```

### Feature Engineering Pipeline

```
Raw Signals → Feature Extraction → Normalization → Model Input
              (73 features)        (StandardScaler)   (73D vector)
```

**Derived Features:**
- Statistical aggregates (mean, median, std, percentiles)
- Temporal patterns (trends, seasonality)
- Cross-feature interactions
- Historical comparison metrics
- Anomaly scores

## Performance and Scalability

### Target SLAs
- **Throughput**: 50,000 requests/second per cluster
- **Latency**: <50ms p95 for verification (inline)
- **Latency**: <100ms p99 for verification
- **Availability**: 99.99% uptime
- **Accuracy**: >97% true positive rate
- **False Positive**: <0.5% (critical for user experience)

### Scaling Strategy

#### Horizontal Scaling Suggestions
```
API Gateway:     10+ instances (auto-scaled)
FastAPI Backend: 50+ pods (HPA based on CPU/latency)
ML Inference:    20+ pods (GPU-accelerated, HPA)
MongoDB:         5-node replica set (sharded by session_id)
Redis:           5-node cluster (master-replica)
Kafka:           7+ brokers (3x replication)
```

#### Vertical Optimization
- **FastAPI**: Async/await throughout, connection pooling
- **ML Inference**: ONNX Runtime for 3-5x speedup
- **Batching**: Micro-batch inference (16-32 requests)
- **Caching**: Redis for 80%+ cache hit rate

### Load Handling Patterns

1. **Rate Limiting** (Multi-layer)
   - Gateway: 1000 req/min per IP
   - Service: 100 req/min per session
   - ML: Token bucket with burst allowance

2. **Circuit Breaker**
   - Trip threshold: 50% error rate
   - Recovery time: 30 seconds
   - Fallback: Allow with logging

3. **Bulkhead Pattern**
   - Separate thread pools per service
   - Isolate ML inference failures
   - Prevent cascade failures

4. **Backpressure Handling**
   - Kafka consumer lag monitoring
   - Dynamic batch size adjustment
   - Graceful degradation to simpler rules

## Security Architecture

### Zero Trust Model

```
┌─────────────────────────────────────────────────────────┐
│  Request Journey                                        │
│  1. Client → Gateway: Verify TLS + request signature    │
│  2. Gateway → Service: mTLS with service identity       │
│  3. Service → MongoDB: IAM role-based authentication    │
│  4. Service → Redis: AUTH + ACLs                        │
│  5. Every hop: Validate, never trust                    │
└─────────────────────────────────────────────────────────┘
```

### Replay Attack Prevention

```python
# Request structure
{
    "session_id": "uuid-v4",
    "timestamp": "unix-epoch-ms",
    "nonce": "random-32-bytes",
    "signals": {encrypted_payload},
    "signature": "HMAC-SHA256(session_key, timestamp + nonce + signals)"
}

# Validation rules
1. Timestamp within ±5 minutes of server time
2. Nonce never seen before (Redis bloom filter)
3. Signature matches HMAC computation
4. Session exists and not expired
```

### Payload Signing

- **Session Establishment**: Client receives ephemeral HMAC key (rotates hourly)
- **Signature Algorithm**: HMAC-SHA256
- **Key Derivation**: HKDF from master secret + session_id
- **Verification**: Constant-time comparison to prevent timing attacks

### Threat Mitigation

| Threat | Mitigation |
|--------|-----------|
| Credential Stuffing | Rate limiting + behavioral anomaly detection |
| Distributed Attacks | IP reputation + device fingerprinting |
| Headless Browsers | WebDriver detection + timing analysis |
| Replay Attacks | Nonce validation + timestamp windows |
| Model Evasion | Online learning + adversarial training |
| Data Poisoning | Outlier detection + human-in-loop validation |

## Privacy and Compliance

### UIDAI/GDPR Alignment

1. **Data Minimization**
   - No biometric collection
   - No personal identifiers
   - No location tracking
   - Only behavioral entropy

2. **Purpose Limitation**
   - Signals used ONLY for bot detection
   - Not shared with third parties
   - Not used for profiling

3. **Short-lived Storage**
   - Raw signals: 24 hours (TTL)
   - Aggregated features: 7 days
   - Verdicts: 30 days (audit requirement)
   - Models: Versioned, no personal data

4. **Transparency**
   - SHAP explanations for every verdict
   - Audit trail for all decisions
   - User visibility into signals collected
   - Right to deletion (session purge)

### Data Retention Policy

```yaml
signals_collection:
  ttl: 24h
  indexes: 
    - session_id (for aggregation)
  
feature_vectors:
  ttl: 7d
  indexes:
    - session_id, timestamp (for drift detection)

verdicts:
  ttl: 30d
  indexes:
    - session_id, timestamp, decision (for audit)

model_artifacts:
  ttl: indefinite
  versioning: semantic versioning
  rollback_capability: 3 previous versions
```

## Adaptive Interaction Strategy

### Decision Tree

```
Inference Result
    ├─> Bot Probability < 0.1 (High confidence human)
    │   └─> ALLOW immediately
    │
    ├─> Bot Probability 0.1 - 0.5 (Uncertain, lean human)
    │   └─> ALLOW + increase monitoring
    │
    ├─> Bot Probability 0.5 - 0.7 (Uncertain, lean bot)
    │   └─> CHALLENGE with adaptive interaction
    │       ├─> Simple task: "Click continue"
    │       ├─> Behavioral test: Time-based interaction
    │       └─> Contextual: Based on user journey
    │
    └─> Bot Probability > 0.7 (High confidence bot)
        └─> BLOCK + log for investigation
```

### Adaptive Challenge Design

**Principles:**
- No explicit puzzles or image selection
- Contextual to user's current task
- Minimal friction (<2 seconds to complete)
- Adaptive difficulty based on suspicion level

**Examples:**
```
Low suspicion (0.5-0.6):
  - "Please click continue to proceed"
  - Analyze click timing and pointer trajectory

Medium suspicion (0.6-0.7):
  - "Please wait 3 seconds before continuing"
  - Analyze attention patterns and micro-movements

High suspicion (0.7-0.8):
  - "Please move your mouse to the button"
  - Analyze natural vs programmatic movement
```

## Observability and Monitoring

### Metrics (Prometheus)

```
# Business Metrics
bot_detection_requests_total{verdict, confidence_range}
bot_detection_accuracy{model_version}
false_positive_rate{threshold}
false_negative_rate{threshold}

# Performance Metrics
verification_latency_seconds{percentile}
ml_inference_latency_seconds{model}
cache_hit_rate{layer}
throughput_requests_per_second{service}

# Health Metrics
service_availability{service}
model_drift_score{model_version}
kafka_consumer_lag{topic}
mongodb_replication_lag{node}
```

### Logging (Structured JSON)

```json
{
  "timestamp": "2025-12-14T10:30:45.123Z",
  "level": "INFO",
  "service": "verification-middleware",
  "trace_id": "7b2f8a3e-9c1d-4f5e-a6b7-3d8e9f0c1a2b",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "event": "verification_completed",
  "verdict": "ALLOW",
  "bot_probability": 0.12,
  "confidence": 0.89,
  "latency_ms": 47,
  "model_version": "v2.3.1",
  "features_count": 73,
  "cache_hit": false
}
```

### Alerting Rules

```yaml
alerts:
  - name: HighFalsePositiveRate
    condition: false_positive_rate > 0.01
    severity: critical
    action: page_on_call
    
  - name: ModelDriftDetected
    condition: drift_score > 0.3
    severity: warning
    action: trigger_retraining
    
  - name: InferenceLatencyHigh
    condition: p95_latency > 100ms
    severity: warning
    action: scale_ml_pods
    
  - name: UnusualBlockRate
    condition: block_rate > 0.05
    severity: warning
    action: investigate_attacks
```

### Distributed Tracing (Jaeger/Tempo)

```
Request flow trace:
├─ api_gateway (5ms)
│  ├─ rate_limit_check (1ms)
│  └─ signature_verify (2ms)
├─ verification_middleware (45ms)
│  ├─ cache_lookup (2ms) [Redis]
│  ├─ ml_inference (35ms)
│  │  ├─ feature_extraction (8ms)
│  │  ├─ model_predict (25ms)
│  │  └─ decision_logic (2ms)
│  └─ verdict_cache (3ms) [Redis]
└─ business_api (150ms)
```

## Deployment Architecture

### Kubernetes Cluster Layout

```yaml
Namespaces:
  - bot-detection-gateway     # API Gateway, ingress controllers
  - bot-detection-api         # FastAPI services
  - bot-detection-ml          # ML inference services
  - bot-detection-streaming   # Kafka, streaming processors
  - bot-detection-data        # MongoDB, Redis operators
  - bot-detection-monitoring  # Prometheus, Grafana, Jaeger
```

### Resource Allocation

```yaml
# High-traffic service (verification-api)
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 8Gi

# ML inference service
resources:
  requests:
    cpu: 4000m
    memory: 8Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 8000m
    memory: 16Gi
    nvidia.com/gpu: 1

# Autoscaling
hpa:
  minReplicas: 10
  maxReplicas: 100
  targetCPU: 70%
  targetMemory: 80%
  custom:
    - type: Pods
      metric: request_latency_p95
      target: 50ms
```

### Multi-Region Deployment

```
Primary Region (India - Mumbai)
├─ Full stack deployment
├─ MongoDB primary
├─ Kafka primary cluster
└─ 70% of traffic

Secondary Region (India - Hyderabad)
├─ Full stack deployment
├─ MongoDB secondary
├─ Kafka follower cluster
└─ 30% of traffic + failover

DR Region (Singapore)
├─ Standby deployment
├─ MongoDB delayed secondary
└─ Activated on regional failure
```

## Failure Scenarios and Mitigations

| Scenario | Detection | Mitigation | RTO |
|----------|-----------|------------|-----|
| ML Service Down | Health check failure | Fallback to rule-based detection | <1 min |
| MongoDB Outage | Connection timeout | Redis cache + read from secondary | <2 min |
| Redis Failure | Connection failure | Direct DB queries + degrade caching | <30 sec |
| Kafka Lag Spike | Consumer lag metric | Increase consumers + drop old events | <5 min |
| Model Accuracy Drop | Drift detection | Rollback to previous model version | <10 min |
| DDoS Attack | Traffic spike + block rate | Activate aggressive rate limiting | Immediate |
| Zero-day Evasion | Anomaly detection | Human review + emergency model update | <1 hour |

## Model Versioning and Rollout

### A/B Testing Strategy

```python
# Route 10% of traffic to new model
if hash(session_id) % 100 < 10:
    model = load_model("v2.4.0-candidate")
else:
    model = load_model("v2.3.1-stable")

# Compare metrics for 48 hours
# If accuracy drop < 1% and latency increase < 10%:
#     Promote v2.4.0 to stable
# Else:
#     Abandon v2.4.0 and investigate
```

### Safe Rollout Process

1. **Canary Deployment** (5% traffic, 4 hours)
2. **Expanded Testing** (25% traffic, 24 hours)
3. **Gradual Rollout** (50% → 75% → 100%, 1 week)
4. **Monitoring Window** (2 weeks close observation)
5. **Rollback Capability** (One-click revert for 30 days)

## Cost Optimization

### Infrastructure Costs (Estimated for 100M requests/day)

```
Kubernetes Cluster: $8,000/month
  - 50 nodes (n2-standard-8): $6,500
  - 10 GPU nodes (n1-standard-8 + T4): $1,500

MongoDB Atlas: $3,000/month
  - M50 replica set (3 nodes)

Redis Enterprise: $1,500/month
  - 5-node cluster, 100GB memory

Kafka (Confluent Cloud): $2,000/month
  - 7 brokers, 1TB storage

Networking & Load Balancing: $1,000/month

Total: ~$15,500/month
Cost per million requests: $4.65
```

### Optimization Strategies
- **Request Batching**: Reduce ML inference calls by 40%
- **Aggressive Caching**: 85% cache hit rate saves 80% of ML costs
- **Spot Instances**: 60% cost reduction for training workloads
- **Auto-scaling**: Scale down during off-peak (50% cost saving)
- **Edge Caching**: CDN reduces backend calls by 30%

## References and Datasets

### Training Data Sources
1. **CSE-CIC-IDS2018**: Network intrusion detection dataset
   - URL: https://www.kaggle.com/datasets/dhoogla/csecicids2018
   - Use: Network-level attack patterns

2. **IDS Intrusion CSV**: General intrusion patterns
   - URL: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
   - Use: Behavioral baselines

3. **Synthetic Behavior Data**: Generated using Selenium + Playwright
   - Human interaction simulations
   - Bot traffic patterns (scrapers, credential stuffers)

4. **Production Feedback Loop**: Real-world data from system
   - Continuously enriches model
   - Captures zero-day attack patterns

### Key Technologies
- **Flutter Web**: Cross-platform frontend
- **FastAPI**: High-performance async Python backend
- **MongoDB**: Flexible document store for signals
- **Redis**: High-speed caching and rate limiting
- **Kafka**: Distributed streaming platform
- **Scikit-learn**: Random Forest implementation
- **XGBoost**: Gradient boosting framework
- **SHAP**: Model explainability
- **Kubernetes**: Container orchestration
- **Prometheus/Grafana**: Monitoring and visualization

---
