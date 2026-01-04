# Project Structure

This document outlines the complete folder structure for the Passive Bot Detection System.

```
passive-bot-detection/
│
├── docs/                                    # Documentation
│   ├── ARCHITECTURE.md                      # System architecture (this file)
│   ├── API_SPECIFICATION.md                 # OpenAPI/Swagger specs
│   ├── DEPLOYMENT_GUIDE.md                  # Deployment instructions
│   ├── SECURITY_REVIEW.md                   # Security analysis
│   ├── PRIVACY_COMPLIANCE.md                # GDPR/UIDAI compliance
│   └── RUNBOOK.md                           # Operational procedures
│
├── frontend/                                # Flutter Web Application
│   ├── lib/
│   │   ├── main.dart                        # Application entry point
│   │   ├── core/
│   │   │   ├── config/
│   │   │   │   └── app_config.dart          # Environment configuration
│   │   │   ├── di/
│   │   │   │   └── injection.dart           # Dependency injection
│   │   │   └── utils/
│   │   │       └── logger.dart              # Logging utility
│   │   ├── features/
│   │   │   └── bot_detection/
│   │   │       ├── data/
│   │   │       │   ├── models/
│   │   │       │   │   ├── signal_payload.dart
│   │   │       │   │   └── verification_response.dart
│   │   │       │   ├── repositories/
│   │   │       │   │   └── bot_detection_repository.dart
│   │   │       │   └── datasources/
│   │   │       │       ├── signal_collector.dart
│   │   │       │       └── api_client.dart
│   │   │       ├── domain/
│   │   │       │   ├── entities/
│   │   │       │   │   └── verification_verdict.dart
│   │   │       │   ├── usecases/
│   │   │       │   │   └── verify_session.dart
│   │   │       │   └── repositories/
│   │   │       │       └── bot_detection_repository_interface.dart
│   │   │       └── presentation/
│   │   │           ├── widgets/
│   │   │           │   └── adaptive_challenge_widget.dart
│   │   │           └── providers/
│   │   │               └── detection_provider.dart
│   │   └── shared/
│   │       ├── widgets/
│   │       └── constants/
│   ├── web/
│   │   ├── index.html
│   │   └── js/
│   │       ├── signal_capture.js           # Browser-level signal capture
│   │       ├── crypto_utils.js             # Payload signing
│   │       └── fingerprint.js              # Device fingerprinting
│   ├── pubspec.yaml
│   └── analysis_options.yaml
│
├── backend/                                 # FastAPI Backend Services
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                          # FastAPI application entry
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py                    # Configuration management
│   │   │   ├── security.py                  # Security utilities
│   │   │   ├── logging.py                   # Structured logging
│   │   │   └── exceptions.py                # Custom exceptions
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── dependencies.py              # FastAPI dependencies
│   │   │   ├── middleware/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── verification_middleware.py  # Main verification logic
│   │   │   │   ├── rate_limiting.py         # Rate limiting middleware
│   │   │   │   ├── request_id.py            # Request ID injection
│   │   │   │   └── correlation_id.py        # Distributed tracing
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── verification.py          # Verification endpoints
│   │   │       ├── health.py                # Health check endpoints
│   │   │       └── admin.py                 # Admin endpoints
│   │   ├── domain/
│   │   │   ├── __init__.py
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── signal.py                # Signal domain model
│   │   │   │   ├── verdict.py               # Verdict domain model
│   │   │   │   └── session.py               # Session domain model
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── verification_service.py  # Core verification logic
│   │   │   │   ├── signal_processor.py      # Signal processing
│   │   │   │   └── decision_engine.py       # Decision logic
│   │   │   └── repositories/
│   │   │       ├── __init__.py
│   │   │       ├── signal_repository.py     # Signal persistence
│   │   │       ├── verdict_repository.py    # Verdict persistence
│   │   │       └── session_repository.py    # Session management
│   │   ├── infrastructure/
│   │   │   ├── __init__.py
│   │   │   ├── database/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── mongodb.py               # MongoDB connection
│   │   │   │   └── redis.py                 # Redis connection
│   │   │   ├── messaging/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── kafka_producer.py        # Kafka producer
│   │   │   │   └── kafka_consumer.py        # Kafka consumer
│   │   │   ├── ml/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── inference_client.py      # ML inference client
│   │   │   │   └── model_loader.py          # Model loading utility
│   │   │   └── cache/
│   │   │       ├── __init__.py
│   │   │       └── verdict_cache.py         # Verdict caching layer
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── signal_schema.py             # Signal Pydantic schemas
│   │       ├── verdict_schema.py            # Verdict Pydantic schemas
│   │       └── verification_schema.py       # Verification request/response
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── load/
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── Dockerfile
│
├── ml-service/                              # ML Inference Service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                          # Inference API entry point
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── inference.py             # Inference endpoints
│   │   │       └── health.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── ensemble.py                  # Ensemble model wrapper
│   │   │   ├── random_forest_wrapper.py     # RF wrapper
│   │   │   ├── xgboost_wrapper.py           # XGBoost wrapper
│   │   │   └── base_model.py                # Base model interface
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── feature_extractor.py         # Feature extraction
│   │   │   ├── feature_transformer.py       # Feature transformation
│   │   │   └── feature_validator.py         # Feature validation
│   │   ├── explainability/
│   │   │   ├── __init__.py
│   │   │   └── shap_explainer.py            # SHAP explanations
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── inference_request.py
│   │       └── inference_response.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml-pipeline/                             # ML Training Pipeline
│   ├── data/
│   │   ├── raw/                             # Raw datasets
│   │   ├── processed/                       # Processed datasets
│   │   └── features/                        # Feature stores
│   ├── notebooks/                           # Jupyter notebooks
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   ├── 03_model_training.ipynb
│   │   └── 04_model_evaluation.ipynb
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── dataset_loader.py            # Load Kaggle datasets
│   │   │   ├── data_validator.py            # Data quality checks
│   │   │   └── data_augmentation.py         # Synthetic data generation
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── feature_engineering.py       # Feature creation
│   │   │   ├── feature_selection.py         # Feature selection
│   │   │   └── feature_store.py             # Feature storage
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── train_random_forest.py       # RF training
│   │   │   ├── train_xgboost.py             # XGBoost training
│   │   │   ├── ensemble_trainer.py          # Ensemble training
│   │   │   └── model_evaluator.py           # Model evaluation
│   │   ├── explainability/
│   │   │   ├── __init__.py
│   │   │   └── shap_analysis.py             # SHAP analysis
│   │   └── deployment/
│   │       ├── __init__.py
│   │       ├── model_packager.py            # Package models
│   │       └── model_registry.py            # MLflow integration
│   ├── configs/
│   │   ├── training_config.yaml
│   │   └── feature_config.yaml
│   ├── scripts/
│   │   ├── train.py                         # Training script
│   │   ├── evaluate.py                      # Evaluation script
│   │   └── export_onnx.py                   # ONNX export
│   ├── requirements.txt
│   └── Dockerfile
│
├── streaming-service/                       # Online Learning Pipeline
│   ├── src/
│   │   ├── __init__.py
│   │   ├── consumers/
│   │   │   ├── __init__.py
│   │   │   ├── signal_consumer.py           # Consume signals
│   │   │   └── verdict_consumer.py          # Consume verdicts
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── drift_detector.py            # Detect model drift
│   │   │   ├── feedback_aggregator.py       # Aggregate feedback
│   │   │   └── anomaly_detector.py          # Detect anomalies
│   │   ├── producers/
│   │   │   ├── __init__.py
│   │   │   ├── training_data_producer.py    # Publish training data
│   │   │   └── alert_producer.py            # Publish alerts
│   │   └── config/
│   │       └── kafka_config.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── infrastructure/                          # Infrastructure as Code
│   ├── kubernetes/
│   │   ├── base/
│   │   │   ├── namespace.yaml
│   │   │   ├── configmap.yaml
│   │   │   └── secrets.yaml.template
│   │   ├── backend/
│   │   │   ├── deployment.yaml              # Backend deployment
│   │   │   ├── service.yaml
│   │   │   ├── hpa.yaml                     # Horizontal Pod Autoscaler
│   │   │   └── pdb.yaml                     # Pod Disruption Budget
│   │   ├── ml-service/
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   └── hpa.yaml
│   │   ├── streaming/
│   │   │   └── deployment.yaml
│   │   ├── data/
│   │   │   ├── mongodb-statefulset.yaml
│   │   │   ├── redis-statefulset.yaml
│   │   │   └── kafka-cluster.yaml
│   │   ├── monitoring/
│   │   │   ├── prometheus-config.yaml
│   │   │   ├── grafana-deployment.yaml
│   │   │   └── alertmanager-config.yaml
│   │   └── ingress/
│   │       ├── api-gateway.yaml
│   │       └── ingress-nginx.yaml
│   ├── terraform/                           # Cloud infrastructure
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── modules/
│   │       ├── vpc/
│   │       ├── eks/
│   │       └── monitoring/
│   └── helm/
│       └── bot-detection/
│           ├── Chart.yaml
│           ├── values.yaml
│           └── templates/
│
├── monitoring/                              # Observability
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alerts.yml
│   ├── grafana/
│   │   └── dashboards/
│   │       ├── bot-detection-overview.json
│   │       ├── ml-performance.json
│   │       └── system-health.json
│   └── jaeger/
│       └── jaeger-config.yaml
│
├── scripts/                                 # Utility scripts
│   ├── setup_dev.sh
│   ├── deploy_staging.sh
│   ├── deploy_production.sh
│   ├── run_load_tests.sh
│   └── backup_models.sh
│
├── .github/
│   └── workflows/
│       ├── backend-ci.yml
│       ├── ml-service-ci.yml
│       ├── frontend-ci.yml
│       └── deploy.yml
│
├── docker-compose.yml                       # Local development
├── Makefile                                 # Common commands
├── .gitignore
├── .env.example
└── README.md
```

## Key Design Principles

### 1. Clean Architecture (Hexagonal)
- **Domain Layer**: Pure business logic, no dependencies
- **Application Layer**: Use cases and orchestration
- **Infrastructure Layer**: External dependencies (DB, Kafka, ML)
- **Presentation Layer**: API endpoints and controllers

### 2. Separation of Concerns
- **Backend**: API orchestration, business rules
- **ML Service**: Inference only, isolated
- **Streaming**: Async processing, decoupled
- **Frontend**: Signal collection, UI

### 3. Scalability Patterns
- **Horizontal Scaling**: All services are stateless
- **Vertical Isolation**: Services can scale independently
- **Data Partitioning**: MongoDB sharding, Kafka partitions

### 4. Observability First
- Structured logging everywhere
- Distributed tracing for all requests
- Metrics for all critical paths
- Alerting for anomalies

### 5. Security by Design
- Zero trust between services
- Payload signing and verification
- Secret management (Kubernetes secrets)
- Audit trail for all decisions

## Technology Justification

### Why FastAPI?
- **Async/Await**: Non-blocking I/O for high concurrency
- **Type Hints**: Strong typing with Pydantic validation
- **Performance**: 3x faster than Flask, close to Go/Node
- **OpenAPI**: Auto-generated API documentation
- **Dependency Injection**: Clean, testable code

### Why MongoDB?
- **Flexible Schema**: Signals structure evolves with attacks
- **TTL Indexes**: Automatic data expiration for privacy
- **Sharding**: Horizontal scaling for billions of documents
- **Geospatial**: Future location-based features
- **Change Streams**: Real-time data pipelines

### Why Redis?
- **Sub-millisecond Latency**: Critical for rate limiting
- **Rich Data Structures**: Sets, sorted sets for complex queries
- **Pub/Sub**: Real-time event broadcasting
- **Persistence**: Optional durability for important caches
- **Cluster Mode**: Horizontal scaling

### Why Kafka?
- **Throughput**: Millions of messages per second
- **Durability**: Persistent message log
- **Replayability**: Reprocess events for training
- **Stream Processing**: Built-in streams API
- **Scalability**: Partitioning and consumer groups

### Why Flutter Web?
- **Single Codebase**: Shared code with mobile apps
- **Performance**: Near-native performance with CanvasKit
- **Accessibility**: Built-in a11y support
- **Security**: Sandboxed execution model
- **Developer Experience**: Hot reload, rich tooling

## Next Steps

1. Implement each component following this structure
2. Set up CI/CD pipelines
3. Deploy to staging environment
4. Load testing and performance tuning
5. Security auditing and penetration testing
6. Production rollout with monitoring
