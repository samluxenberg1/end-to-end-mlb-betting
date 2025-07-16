from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ModelConfig:
    """Configure class for model training parameters"""
    model_type: str
    hyperparams: Dict[str, Any]
    cv_params: Dict[str, Any]
    categorical_cols: List[str]
    encoding_type: str = 'one-hot'
    calibration_method: str = 'isotonic' # or sigmoid
    random_state: int = 888

@dataclass
class TrainingResults:
    """Container for training results and metrics"""
    fold_accuracies: List[float]
    fold_log_losses: List[float]
    fold_brier_scores: List[float]
    fold_auc_scores: List[float]
    calibration_scores: List[float]
    mean_accuracy: float
    std_accuracy: float
    mean_log_loss: float
    mean_brier_score: float
    mean_auc: float
    mean_calibration_score: float