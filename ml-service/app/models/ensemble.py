"""
Ensemble Bot Detection Model

Combines Random Forest and XGBoost for robust bot detection.
Achieves ~97% accuracy based on MVP results.

Architecture:
- Random Forest: Base model, robust to outliers
- XGBoost: Boosted model, handles imbalanced data
- Weighted ensemble: Dynamic weighting based on confidence
"""

import numpy as np
import joblib
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnsembleVerificationModel:
    """
    Ensemble model for bot detection

    Combines Random Forest and XGBoost predictions with confidence-based weighting.
    Supports both scikit-learn/xgboost and ONNX runtime for faster inference.
    """

    def __init__(
        self,
        rf_model_path: Optional[str] = None,
        xgb_model_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        model_version: str = "1.0.0",
        use_onnx: bool = True,
    ):
        """
        Initialize ensemble model

        Args:
            rf_model_path: Path to Random Forest model (pkl or onnx)
            xgb_model_path: Path to XGBoost model (pkl or onnx)
            feature_names: List of expected feature names
            model_version: Model version identifier
            use_onnx: Use ONNX runtime if available (3-5x faster)
        """
        self.model_version = model_version
        self.feature_names = feature_names or []
        self.use_onnx = use_onnx and ONNX_AVAILABLE

        # Model weights (can be tuned based on validation performance)
        self.rf_base_weight = 0.4
        self.xgb_base_weight = 0.6

        # Confidence thresholds
        self.high_confidence_threshold = 0.85
        self.low_confidence_threshold = 0.6

        # Load models
        self.rf_model = None
        self.xgb_model = None
        self.rf_onnx_session = None
        self.xgb_onnx_session = None

        if rf_model_path:
            self.load_rf_model(rf_model_path)

        if xgb_model_path:
            self.load_xgb_model(xgb_model_path)

        logger.info(
            f"Ensemble model initialized",
            extra={
                "version": model_version,
                "use_onnx": self.use_onnx,
                "rf_loaded": self.rf_model is not None
                or self.rf_onnx_session is not None,
                "xgb_loaded": self.xgb_model is not None
                or self.xgb_onnx_session is not None,
            },
        )

    def load_rf_model(self, model_path: str) -> None:
        """Load Random Forest model"""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"RF model not found: {model_path}")

        if self.use_onnx and model_path.suffix == ".onnx":
            # Load ONNX model
            self.rf_onnx_session = ort.InferenceSession(
                str(model_path),
                providers=[
                    "CPUExecutionProvider"
                ],  # Use GPU if available: 'CUDAExecutionProvider'
            )
            logger.info(f"Random Forest ONNX model loaded from {model_path}")
        else:
            # Load scikit-learn model
            self.rf_model = joblib.load(model_path)
            logger.info(f"Random Forest sklearn model loaded from {model_path}")

    def load_xgb_model(self, model_path: str) -> None:
        """Load XGBoost model"""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")

        if self.use_onnx and model_path.suffix == ".onnx":
            # Load ONNX model
            self.xgb_onnx_session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            logger.info(f"XGBoost ONNX model loaded from {model_path}")
        else:
            # Load XGBoost model
            self.xgb_model = joblib.load(model_path)
            logger.info(f"XGBoost model loaded from {model_path}")

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction for a single sample

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Dictionary containing:
                - bot_probability: Probability that user is a bot [0-1]
                - confidence: Model confidence in prediction [0-1]
                - decision: Recommended action (allow/challenge/block)
                - rf_vote: Random Forest prediction
                - xgb_vote: XGBoost prediction
                - model_version: Version identifier
        """
        # Convert features dict to numpy array in correct order
        feature_vector = self._prepare_features(features)

        # Get predictions from both models
        rf_proba = self._predict_rf(feature_vector)
        xgb_proba = self._predict_xgb(feature_vector)

        # Compute confidence for each model
        rf_confidence = self._compute_confidence(rf_proba)
        xgb_confidence = self._compute_confidence(xgb_proba)

        # Dynamic weighting based on confidence
        rf_weight, xgb_weight = self._compute_weights(rf_confidence, xgb_confidence)

        # Weighted ensemble
        final_proba = rf_weight * rf_proba[1] + xgb_weight * xgb_proba[1]
        final_confidence = (rf_confidence + xgb_confidence) / 2

        # Determine decision
        decision = self._make_decision(final_proba, final_confidence)

        return {
            "bot_probability": float(final_proba),
            "confidence": float(final_confidence),
            "decision": decision,
            "rf_vote": float(rf_proba[1]),
            "xgb_vote": float(xgb_proba[1]),
            "rf_confidence": float(rf_confidence),
            "xgb_confidence": float(xgb_confidence),
            "rf_weight": float(rf_weight),
            "xgb_weight": float(xgb_weight),
            "model_version": self.model_version,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def predict_batch(
        self, features_list: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of samples (more efficient)

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of prediction dictionaries
        """
        # Convert to feature matrix
        feature_matrix = np.array(
            [self._prepare_features(features) for features in features_list]
        )

        # Batch predictions
        rf_proba_batch = self._predict_rf_batch(feature_matrix)
        xgb_proba_batch = self._predict_xgb_batch(feature_matrix)

        # Process each prediction
        results = []
        for i in range(len(features_list)):
            rf_proba = rf_proba_batch[i]
            xgb_proba = xgb_proba_batch[i]

            rf_confidence = self._compute_confidence(rf_proba)
            xgb_confidence = self._compute_confidence(xgb_proba)

            rf_weight, xgb_weight = self._compute_weights(rf_confidence, xgb_confidence)

            final_proba = rf_weight * rf_proba[1] + xgb_weight * xgb_proba[1]
            final_confidence = (rf_confidence + xgb_confidence) / 2

            decision = self._make_decision(final_proba, final_confidence)

            results.append(
                {
                    "bot_probability": float(final_proba),
                    "confidence": float(final_confidence),
                    "decision": decision,
                    "rf_vote": float(rf_proba[1]),
                    "xgb_vote": float(xgb_proba[1]),
                    "model_version": self.model_version,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        return results

    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to numpy array in correct order

        Handles missing features by filling with 0 (or configured default)
        """
        if not self.feature_names:
            # No ordering specified, use dict order (dangerous in production)
            logger.warning("No feature names specified - using arbitrary order")
            return np.array(list(features.values()), dtype=np.float32)

        # Ensure features are in correct order
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                # Missing feature - fill with 0
                logger.warning(f"Missing feature: {feature_name}, filling with 0")
                feature_vector.append(0.0)

        return np.array(feature_vector, dtype=np.float32)

    def _predict_rf(self, feature_vector: np.ndarray) -> np.ndarray:
        """Get Random Forest prediction probabilities"""
        if self.rf_onnx_session:
            # ONNX inference
            input_name = self.rf_onnx_session.get_inputs()[0].name
            output_name = self.rf_onnx_session.get_outputs()[0].name

            # Reshape for batch inference
            features = feature_vector.reshape(1, -1).astype(np.float32)
            proba = self.rf_onnx_session.run([output_name], {input_name: features})[0][
                0
            ]
            return proba

        elif self.rf_model:
            # sklearn inference
            proba = self.rf_model.predict_proba(feature_vector.reshape(1, -1))[0]
            return proba

        else:
            raise RuntimeError("Random Forest model not loaded")

    def _predict_xgb(self, feature_vector: np.ndarray) -> np.ndarray:
        """Get XGBoost prediction probabilities"""
        if self.xgb_onnx_session:
            # ONNX inference
            input_name = self.xgb_onnx_session.get_inputs()[0].name
            output_name = self.xgb_onnx_session.get_outputs()[0].name

            features = feature_vector.reshape(1, -1).astype(np.float32)
            proba = self.xgb_onnx_session.run([output_name], {input_name: features})[0][
                0
            ]
            return proba

        elif self.xgb_model:
            # XGBoost inference
            import xgboost as xgb

            dmatrix = xgb.DMatrix(feature_vector.reshape(1, -1))
            proba = self.xgb_model.predict(dmatrix)[0]

            # XGBoost might return single value or array
            if isinstance(proba, (int, float)):
                return np.array([1 - proba, proba])
            return proba

        else:
            raise RuntimeError("XGBoost model not loaded")

    def _predict_rf_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Batch Random Forest prediction"""
        if self.rf_onnx_session:
            input_name = self.rf_onnx_session.get_inputs()[0].name
            output_name = self.rf_onnx_session.get_outputs()[0].name
            return self.rf_onnx_session.run(
                [output_name], {input_name: feature_matrix.astype(np.float32)}
            )[0]
        elif self.rf_model:
            return self.rf_model.predict_proba(feature_matrix)
        else:
            raise RuntimeError("Random Forest model not loaded")

    def _predict_xgb_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Batch XGBoost prediction"""
        if self.xgb_onnx_session:
            input_name = self.xgb_onnx_session.get_inputs()[0].name
            output_name = self.xgb_onnx_session.get_outputs()[0].name
            return self.xgb_onnx_session.run(
                [output_name], {input_name: feature_matrix.astype(np.float32)}
            )[0]
        elif self.xgb_model:
            import xgboost as xgb

            dmatrix = xgb.DMatrix(feature_matrix)
            probas = self.xgb_model.predict(dmatrix)

            # Ensure 2D array [n_samples, 2]
            if len(probas.shape) == 1:
                probas = np.column_stack([1 - probas, probas])
            return probas
        else:
            raise RuntimeError("XGBoost model not loaded")

    def _compute_confidence(self, proba: np.ndarray) -> float:
        """
        Compute model confidence from probability distribution

        Confidence is based on how close the prediction is to 0 or 1.
        A prediction of 0.5 has low confidence (uncertain).
        A prediction of 0.95 has high confidence.
        """
        # Confidence = 2 * |max(proba) - 0.5|
        # This gives 0 for proba=0.5, and 1 for proba=0 or proba=1
        max_proba = np.max(proba)
        confidence = 2 * abs(max_proba - 0.5)
        return confidence

    def _compute_weights(
        self, rf_confidence: float, xgb_confidence: float
    ) -> Tuple[float, float]:
        """
        Compute dynamic weights based on model confidence

        If one model is more confident, give it more weight.
        """
        total_confidence = rf_confidence + xgb_confidence

        if total_confidence > 0:
            # Confidence-weighted
            rf_weight = rf_confidence / total_confidence
            xgb_weight = xgb_confidence / total_confidence
        else:
            # Fallback to base weights
            rf_weight = self.rf_base_weight
            xgb_weight = self.xgb_base_weight

        return rf_weight, xgb_weight

    def _make_decision(self, bot_probability: float, confidence: float) -> str:
        """
        Make allow/challenge/block decision

        This is a recommendation - final decision is made by middleware
        based on additional context and configuration.
        """
        if confidence < self.low_confidence_threshold:
            return "uncertain"

        if bot_probability < 0.1:
            return "allow"
        elif bot_probability < 0.7:
            return "monitor"
        elif bot_probability < 0.9:
            return "challenge"
        else:
            return "block"

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from models

        Returns combined feature importance from both models
        """
        importance = {}

        # Random Forest importance
        if self.rf_model and hasattr(self.rf_model, "feature_importances_"):
            rf_importance = self.rf_model.feature_importances_
            for i, feature_name in enumerate(self.feature_names):
                importance[f"rf_{feature_name}"] = float(rf_importance[i])

        # XGBoost importance
        if self.xgb_model and hasattr(self.xgb_model, "feature_importances_"):
            xgb_importance = self.xgb_model.feature_importances_
            for i, feature_name in enumerate(self.feature_names):
                importance[f"xgb_{feature_name}"] = float(xgb_importance[i])

        return importance

    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate that features are complete and valid

        Returns:
            (is_valid, list of issues)
        """
        issues = []

        # Check for missing features
        missing_features = set(self.feature_names) - set(features.keys())
        if missing_features:
            issues.append(f"Missing features: {missing_features}")

        # Check for extra features (warning, not error)
        extra_features = set(features.keys()) - set(self.feature_names)
        if extra_features:
            issues.append(f"Unexpected features (will be ignored): {extra_features}")

        # Check for invalid values (NaN, inf)
        for name, value in features.items():
            if not isinstance(value, (int, float)):
                issues.append(f"Feature {name} has non-numeric value: {type(value)}")
            elif np.isnan(value) or np.isinf(value):
                issues.append(f"Feature {name} has invalid value: {value}")

        return len(issues) == 0, issues
