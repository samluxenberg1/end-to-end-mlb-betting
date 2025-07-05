import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, 
    log_loss, 
    brier_score_loss,
    roc_auc_score,
    classification_report
)
from sklearn.calibration import calibration_curve
from mlxtend.evaluate.time_series import GroupTimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import pickle
import joblib
from datetime import datetime
import warnings

from models.archive.utils_models_v1 import (
    load_model_data, 
    data_split,
    categorical_encoding
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for model training parameters"""
    hyperparams: Dict[str, Any]
    cv_params: Dict[str, Any]
    categorical_cols: List[str]
    encoding_type: str = "one-hot"
    calibration_method: str = "isotonic" # or sigmoid
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

class MLBModelTrainer:
    """
    Production-ready MLB model trainer with calibration capabilities.

    Features: 
    - Time series cross-validation with proper data leakage prevention
    - Probability calibration (isotonic or sigmoid)
    - Comprehensive evaluation metrics
    - Calibration curve plotting
    - Model persistence
    - Logging and error handling
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.calibrated_model = None
        self.results = None
        self.feature_names = None
        self.is_trained = False

    def _validate_data(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> None:
        """Validate input data for training."""
        if X.empty or y.empty:
            raise ValueError("Training data cannot be empty")
        
        if (len(X) != len(y)) or (len(X) != len(groups)):
            raise ValueError("X, y, and groups must have the same length")
        
        if y.isnull().any():
            raise ValueError("Target variable contains null values")
        
        # Check for required categorical columns
        missing_cols = set(self.config.categorical_cols) - set(X.columns)
        if missing_cols: 
            raise ValueError("Missing required categorical columns: {missing_cols}")
        
    def _calculate_calibration_score(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate calibration score (reliability)"""
        try: 
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy="uniform"
            )
            # Calculate Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0,1,n_bins+1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
            return ece
        
        except Exception as e:
            logger.warning(f"Could not calculation calibration score: {e}")
            
            return np.nan
    
    def train_with_cross_validation(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> TrainingResults:
        """
        Train model with time series cross-validation and calibration.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        groups : pd.Series
            Time-based groups for cross-validation

        Returns
        -------
        TrainingResults
            Training results object with comprehensive metrics
        """
        self._validate_data(X, y, groups)

        # Store feature names for later use
        self.feature_names = X.columns.tolist()

        # Initialize cross-validation
        gts = GroupTimeSeriesSplit(**self.config.cv_params)

        # Initialize base model
        base_model = RandomForestClassifier(**self.config.hyperparams)

        # Storage for results
        fold_results = {
            'accuracies': [],
            'log_losses': [],
            'brier_scores': [],
            'auc_scores': [],
            'calibration_scores': []
        }

        logger.info("Starting cross-validation training...")

        fold_num = 0
        for train_index, test_index in gts.split(X, y, groups=groups):
            fold_num += 1

            try: 
                # Extract training and testing sets for this fold
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Get the unique dates for current train and test sets
                train_dates = groups.iloc[train_index].unique()
                test_dates = groups.iloc[test_index].unique()

                logger.info(f"Fold {fold_num}")
                logger.info(f"  Train samples: {len(X_train)} | Test samples: {len(X_test)}")
                logger.info(f"  Train dates: {len(train_dates)} ({train_dates.min()} to {train_dates.max()})")
                logger.info(f"  Test dates: {len(test_dates)} ({test_dates.min()} to {test_dates.max()})")

                # Verify no date overlap
                if len(train_dates) > 0 and len(test_dates) > 0:
                    if train_dates.max() >= test_dates.min():
                        logger.error(f"Date overlap detected! Train max: {train_dates.max()}, Test min: {test_dates.min()}")
                        raise ValueError("Date overlap detected in cross-validation split")
                    
                
                # Apply categorical encoding
                X_train_encoded, X_test_encoded = categorical_encoding(
                    X_train=X_train,
                    X_test=X_test,
                    categorical_cols=self.config.categorical_cols,
                    encoding_type=self.config.encoding_type
                )

                # Create calibrated classifier
                calibrated_model = CalibratedClassifierCV(
                    base_model, 
                    method=self.config.calibration_method,
                    cv=3 # internal CV for calibration
                )
                
                # Train calibrated model
                calibrated_model.fit(X_train_encoded, y_train)

                # Make predictions
                y_pred = calibrated_model.predict(X_test_encoded)
                y_pred_proba = calibrated_model.predict_proba(X_test_encoded)[:, 1]

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                log_loss_score = log_loss(y_test, y_pred_proba)
                brier_score = brier_score_loss(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                calibration_score = self._calculate_calibration_score(y_test, y_pred_proba)

                # Store results
                fold_results['accuracies'].append(accuracy)
                fold_results['log_losses'].append(log_loss_score)
                fold_results['brier_scores'].append(brier_score)
                fold_results['auc_scores'].append(auc_score)
                fold_results['calibration_scores'].append(calibration_score)

                logger.info(f"  Accuracy: {accuracy: .4f}")
                logger.info(f"  Log Loss: {log_loss_score: .4f}")
                logger.info(f"  Brier Score: {brier_score: .4f}")
                logger.info(f"  AUC: {auc_score: .4f}")
                logger.info(f"  Calibration Score (ECE): {calibration_score: .4f}")

            except Exception as e:
                logger.error(f"Error in fold {fold_num} as e")
                raise 
        
        # Calculate summary statistics
        results = TrainingResults(
            fold_accuracies=fold_results['accuracies'],
            fold_log_losses=fold_results['log_losses'],
            fold_brier_scores=fold_results['brier_scores'],
            fold_auc_scores=fold_results['auc_scores'],
            calibration_scores=fold_results['calibration_scores'],
            mean_accuracy=np.mean(fold_results['accuracies']),
            std_accuracy=np.std(fold_results['accuracies']),
            mean_log_loss=np.mean(fold_results['log_losses']),
            mean_brier_score=np.mean(fold_results['brier_scores']),
            mean_auc=np.mean(fold_results['auc_scores']),
            mean_calibration_score=np.mean([s for s in fold_results['calibration_scores'] if not np.isnan(s)])
        )

        logger.info("\n" + "="*50)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Mean Accuracy: {results.mean_accuracy: .4f} (+/-{results.std_accuracy: .4f})")
        logger.info(f"Mean Log Loss: {results.mean_log_loss:.4f}")
        logger.info(f"Mean Brier Score: {results.mean_brier_score:.4f}")
        logger.info(f"Mean AUC: {results.mean_auc:.4f}")
        logger.info(f"Mean Calibration Score (ECE): {results.mean_calibration_score:.4f}")

        self.results = results

        return results
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the final model on all available data"""
        logger.info("Training final calibrated model on all data...")

        # Apply categorical encoding
        logger.info(f"Columns of X: {X.columns}")
        X_encoded, _ = categorical_encoding(
            X_train=X, 
            X_test=pd.DataFrame(), # empty test set
            categorical_cols=self.config.categorical_cols,
            encoding_type=self.config.encoding_type
        )

        # Create and train final model
        base_model = RandomForestClassifier(**self.config.hyperparams)
        self.calibrated_model = CalibratedClassifierCV(
            base_model,
            method=self.config.calibration_method,
            cv=3
        )

        self.calibrated_model.fit(X_encoded, y)
        self.is_trained = True

        logger.info("Final model training completed")

    def plot_calibration_curve(self, X: pd.DataFrame, y: pd.Series, save_path: Optional[str] = None) -> None:
        """Plot calibration curve for the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting calibration curve")
        
        # Apply categorical encoding
        X_encoded, _ = categorical_encoding(
            X_train=X, 
            X_test=pd.DataFrame(),
            categorical_cols=self.config.categorical_cols, 
            encoding_type=self.config.encoding_type
        )

        # Get probabilities
        y_pred_proba = self.calibrated_model.predict_proba(X_encoded)[:, 1]

        # Creat calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, y_pred_proba, n_bins=10
        )

        # Plot
        plt.figure(figsize=(10,6))

        # Calibration Curve
        plt.subplot(1, 2, 1)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated RF")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True)
        
        # Histogram of probabilities
        plt.subplot(1, 2, 2)
        plt.hist(y_pred_proba, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.title("Distribution of Predicted Probabilities")
        plt.grid(True)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration of Predicted Probabilities")
        
        plt.show()

    def save_model(self, file_path: str) -> None: 
        """Save the trained model and configuration"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.calibrated_model,
            'config': self.config,
            'results': self.results,
            'feature_names': self.feature_names,
            'training_timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")

    @classmethod
    def load_models(cls, file_path: str) -> 'MLBModelTrainer':
        """Load a saved model"""
        model_data = joblib.load(file_path)

        trainer = cls(model_data['config'])
        trainer.calibrated_model = model_data['model']
        trainer.results = model_data['results']
        trainer.feature_names = model_data['feature_names']
        trainer.is_trained = True

        logger.info(f"Model loaded from {file_path}")

        return trainer
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model"""
        if not self.is_trained: 
            raise ValueError("Model must be trained before making predictions")
        
        # Apply categorical encoding
        X_encoded, _ = categorical_encoding(
            X_train=X,
            X_test=pd.DataFrame(),
            categorical_cols=self.config.categorical_cols,
            encoding_type=self.config.encoding_type
        )

        return self.calibrated_model.predict(X_encoded)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get calibrated probabilities from the trained model"""
        if not self.is_trained: 
            raise ValueError("Model must be trained before making predictions")
        
        # Apply categorical encoding
        X_encoded, _ = categorical_encoding(
            X_train=X,
            X_test=pd.DataFrame(),
            categorical_cols=self.config.categorical_cols,
            encoding_type=self.config.encoding_type
        )

        return self.calibrated_model.predict_proba(X_encoded)
    

if __name__=='__main__':

    # Load data
    input_model_data = "data/processed/model_data.csv"
    df_model = load_model_data(input_model_data)
    target = 'home_win'

    # Data splitting
    df_train_init, df_holdout, _ = data_split(
        df=df_model,
        holdout_start_date='2025-05-01',
        group_col='game_date'
    )

    # Prepare target variables
    y_train_init = df_train_init[target]
    y_holdout = df_holdout[target]

    # Define groups for GroupTSCV - do i not need groups from data_split() ? --> will groups for holdout be useful?
    groups = df_train_init['game_date']

    # Remove columns
    cols_to_remove = [
        "game_id", "game_date", "game_date_time", "home_team_id",
        "away_team_id", "home_score", "away_score", "state", target
    ]

    df_train_init = df_train_init.drop(cols_to_remove, axis=1)
    df_holdout = df_holdout.drop(cols_to_remove, axis=1)

    # Configuration
    config = ModelConfig(
        hyperparams={
            'min_samples_leaf': 5, 
            'class_weight': 'balanced',
            'random_state': 888,
            'n_jobs': -1
        },
        cv_params={
            'test_size': 30,
            #'train_size': 360,
            'n_splits': 3, # for testing purposes
            'gap_size': 3,
            'window_type': 'rolling'
        },
        categorical_cols=['home_team','away_team','venue','game_type'],
        encoding_type='one-hot',
        calibration_method='isotonic'
    )

    # Train model
    trainer = MLBModelTrainer(config)

    # Cross-validation
    results = trainer.train_with_cross_validation(df_train_init, y_train_init, groups)

    # Train final model
    trainer.train_final_model(df_train_init, y_train_init)

    # Plot calibration curve
    trainer.plot_calibration_curve(df_train_init, y_train_init, save_path='calibration_curve.png')
    trainer.plot_calibration_curve(df_holdout, y_holdout, save_path='calibration_curve_test.png')

    # Save model
    trainer.save_model('trained_mlb_model.pkl')
    # Evaluate on holdout set
    holdout_predictions = trainer.predict(df_holdout)
    holdout_probabilities = trainer.predict_proba(df_holdout)[:, 1]

    holdout_accuracy = accuracy_score(y_holdout, holdout_predictions)
    holdout_log_loss = log_loss(y_holdout, holdout_probabilities)
    holdout_brier = brier_score_loss(y_holdout, holdout_probabilities)
    holdout_auc = roc_auc_score(y_holdout, holdout_probabilities)

    logger.info("\n" + "="*50)
    logger.info("HOLDOUT SET EVALUTION")
    logger.info("="*50)
    logger.info(f"Holdout Accuracy: {holdout_accuracy: .4f}")
    logger.info(f"Holdout Log Loss: {holdout_log_loss: .4f}")
    logger.info(f"Holdout Brier Score: {holdout_brier: .4f}")
    logger.info(f"Holdout AUC: {holdout_auc: .4f}")
    




                
