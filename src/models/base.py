"""
base.py — standard model interface and shared data structures.

All models (classical and DL) must conform to ModelInterface so they
can be composed in the pipeline layer without model-specific branching.

PredictionResult
    Dataclass for a single engine's RUL prediction with uncertainty bounds.
    Classical models: bounds from SARIMAX conf_int.
    DL point models:  bounds from MC Dropout.
    DL quantile models: bounds = (Q10, Q90).

ModelInterface
    Abstract base class.  Both classical wrappers (ClassicalRULModel) and
    DL model wrappers (DLRULModel) implement this interface, enabling the
    pipeline to call fit() / predict() uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PredictionResult:
    """
    Unified output format for a single engine prediction.

    Fields
    ------
    engine_id        : int   — engine identifier
    rul_pred         : float — point-prediction RUL (median / mean)
    lower_bound      : float — pessimistic RUL (upper CI / Q10 path)
    upper_bound      : float — optimistic  RUL (lower CI / Q90 path)
    confidence_width : float — upper_bound - lower_bound (0 = no uncertainty)
    model_name       : str   — e.g. "AR(2)", "ARIMA(1,2,2)", "Q_Transformer"

    Convention (safety-critical)
    ----------------------------
    lower_bound is the EARLIEST predicted failure (most conservative warning).
    upper_bound is the LATEST  predicted failure (most optimistic estimate).
    A maintenance engineer should schedule service by lower_bound to guarantee
    the engine does not fail within the prediction interval.
    """

    engine_id:        int
    rul_pred:         float
    lower_bound:      float
    upper_bound:      float
    confidence_width: float
    model_name:       str

    def __post_init__(self):
        if self.lower_bound > self.upper_bound + 1e-6:
            raise ValueError(
                f"engine {self.engine_id}: lower_bound ({self.lower_bound:.1f}) "
                f"> upper_bound ({self.upper_bound:.1f})"
            )

    def to_dict(self) -> dict:
        return {
            "engine_id":        self.engine_id,
            "rul_pred":         self.rul_pred,
            "lower_bound":      self.lower_bound,
            "upper_bound":      self.upper_bound,
            "confidence_width": self.confidence_width,
            "model_name":       self.model_name,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INTERFACE
# ══════════════════════════════════════════════════════════════════════════════


class ModelInterface(ABC):
    """
    Abstract base for all RUL prediction models.

    Concrete subclasses: ClassicalRULModel, DLPointModel, DLQuantileModel.

    The pipeline layer (src/pipeline/) calls only fit() and predict(),
    ensuring models are interchangeable without branching.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name, e.g. 'ARIMA(1,2,2)' or 'Q_Transformer'."""

    @abstractmethod
    def fit(self, train_data, **kwargs) -> "ModelInterface":
        """
        Fit the model on training data.

        Parameters
        ----------
        train_data : model-specific format
            Classical: pd.DataFrame with health_index + RUL columns.
            DL:        (X_train, y_train) numpy arrays.

        Returns self for chaining.
        """

    @abstractmethod
    def predict(self, test_data, **kwargs) -> list[PredictionResult]:
        """
        Generate predictions for all engines in test_data.

        Parameters
        ----------
        test_data : model-specific format
            Classical: pd.DataFrame (one row per cycle, grouped by engine_id).
            DL:        X_test numpy array (n_engines, window, features).

        Returns
        -------
        list of PredictionResult, one per engine.
        """

    def predict_single(self, series_or_window, engine_id: int = -1, **kwargs) -> PredictionResult:
        """
        Convenience method for a single engine.  Default implementation
        wraps predict() — override for efficiency when needed.
        """
        results = self.predict(series_or_window, **kwargs)
        if results:
            r = results[0]
            return PredictionResult(
                engine_id=engine_id,
                rul_pred=r.rul_pred,
                lower_bound=r.lower_bound,
                upper_bound=r.upper_bound,
                confidence_width=r.confidence_width,
                model_name=self.name,
            )
        raise RuntimeError(f"{self.name}.predict() returned empty list")
