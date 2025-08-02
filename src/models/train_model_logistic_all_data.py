import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, 
    log_loss, 
    brier_score_loss,
    roc_auc_score,
    classification_report
)
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import OneHotEncoder
from mlxtend.evaluate.time_series import GroupTimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
import pickle
import joblib
from datetime import datetime
import warnings

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

#import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA

from src.models.utils_models import (
    load_model_data, 
    data_split,
    transform_categorical_features
)
from src.models.model_classes import ModelConfig, TrainingResults
from src.gen_utils import load_config
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.one_hot_encoder = None # Stores fitted OneHotEncoder
        self.category_maps = None # Stores dictionary of {col: [categories]} for 'category' dtype
        self.numerical_feature_names = None # Stores names of numerical features
        self.all_final_feature_names = None # Stores the complete ordered list of all feature names after encoding

    #def _get_lgbm_model(self):
    #    """Initializes a LightGBM Classifier model with hyperparameters"""
    #    lgbm_parameters = self.config.hyperparams.copy()
    #    if self.config.encoding_type == "category":
    #        lgbm_parameters['categorical_feature'] = self.config.categorical_cols
    #    return lgb.LGBMClassifier(**lgbm_parameters)

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
        # Data validation checks
        self._validate_data(X, y, groups)

        # Initialize cross-validation
        gts = GroupTimeSeriesSplit(**self.config.cv_params)

        # Initialize base model
        #base_model = cb.CatBoostClassifier(**self.config.hyperparams)
        base_model = LogisticRegression(**self.config.hyperparams)

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
                X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
                y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

                # Get the unique dates for current train and test sets
                train_dates = groups.iloc[train_index].unique()
                test_dates = groups.iloc[test_index].unique()

                logger.info(f"Fold {fold_num}")
                logger.info(f"  Train samples: {len(X_train_fold)} | Test samples: {len(X_test_fold)}")
                logger.info(f"  Train dates: {len(train_dates)} ({train_dates.min()} to {train_dates.max()})")
                logger.info(f"  Test dates: {len(test_dates)} ({test_dates.min()} to {test_dates.max()})")

                # Verify no date overlap
                if len(train_dates) > 0 and len(test_dates) > 0:
                    if train_dates.max() >= test_dates.min():
                        logger.error(f"Date overlap detected! Train max: {train_dates.max()}, Test min: {test_dates.min()}")
                        raise ValueError("Date overlap detected in cross-validation split")
                    
                
                # --- Categorical Encoding for CV Fold ---
                # For CV, re-fit the encoder/category maps for each fold
                # to simulate a growing knowledge base over time.
                fold_numerical_cols = [col for col in X_train_fold if col not in self.config.categorical_cols]

                fold_one_hot_encoder = None
                fold_category_maps = None
                fold_all_final_cols = None

                if self.config.encoding_type == "one-hot":
                    fold_one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    fold_one_hot_encoder.fit(X_train_fold[self.config.categorical_cols])
                    fold_encoded_cat_names = fold_one_hot_encoder.get_feature_names_out(self.config.categorical_cols)
                    fold_all_final_cols = fold_numerical_cols + list(fold_encoded_cat_names)
                elif self.config.encoding_type == "category":
                    fold_category_maps = {col: X_train_fold[col].unique().tolist() for col in self.config.categorical_cols}
                    fold_all_final_cols = fold_numerical_cols + self.config.categorical_cols

                X_train_encoded_fold = transform_categorical_features(
                    df=X_train_fold,
                    categorical_cols=self.config.categorical_cols,
                    encoding_type=self.config.encoding_type,
                    one_hot_encoder=fold_one_hot_encoder,
                    category_maps=fold_category_maps,
                    numerical_cols=fold_numerical_cols,
                    all_final_cols=fold_all_final_cols
                )

                X_test_encoded_fold = transform_categorical_features(
                    df=X_test_fold,
                    categorical_cols=self.config.categorical_cols,
                    encoding_type=self.config.encoding_type,
                    one_hot_encoder=fold_one_hot_encoder,
                    category_maps=fold_category_maps,
                    numerical_cols=fold_numerical_cols,
                    all_final_cols=fold_all_final_cols
                )
                # --- End Categorical Encoding for CV Fold ---

                # Store feature names from the *first* fold's processed training data.
                # These will be the definitive feature names for the final model and prediction.
                # Assumes consistent feature generation across folds based on configuration.
                if self.all_final_feature_names is None:
                    self.numerical_feature_names = fold_numerical_cols
                    self.all_final_feature_names = fold_all_final_cols
                    self.feature_names = fold_all_final_cols
                
                # Fit Logistic Regression
                base_model.fit(X_train_encoded_fold, y_train_fold)

                # Create calibrated classifier
                calibrated_model = CalibratedClassifierCV(
                    estimator=FrozenEstimator(base_model), 
                    method=self.config.calibration_method
                )
                
                # Train calibrated model
                calibrated_model.fit(X_train_encoded_fold, y_train_fold)

                # Make predictions
                y_pred = calibrated_model.predict(X_test_encoded_fold)
                y_pred_proba = calibrated_model.predict_proba(X_test_encoded_fold)[:, 1]

                # Calculate metrics
                accuracy = accuracy_score(y_test_fold, y_pred)
                log_loss_score = log_loss(y_test_fold, y_pred_proba)
                brier_score = brier_score_loss(y_test_fold, y_pred_proba)
                auc_score = roc_auc_score(y_test_fold, y_pred_proba)
                calibration_score = self._calculate_calibration_score(y_test_fold, y_pred_proba)

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

                #mlflow.log_metric(f"fold_{fold_num}_accuracy", accuracy)
                #mlflow.log_metric(f"fold_{fold_num}_log_loss", log_loss_score)
                #mlflow.log_metric(f"fold_{fold_num}_brier_score", brier_score)
                #mlflow.log_metric(f"fold_{fold_num}_auc_score", auc_score)
                #if not np.isnan(calibration_score):
                #    mlflow.log_metric(f"fold_{fold_num}_calibration_score", calibration_score)

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

        # Store numerical feature names
        self.numerical_feature_names = [col for col in X.columns if col not in self.config.categorical_cols]

        # Fit and store the appropriate encoder/category maps based on encoding_type
        if self.config.encoding_type == "one-hot":
            self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.one_hot_encoder.fit(X[self.config.categorical_cols])
            encoded_cat_names = self.one_hot_encoder.get_feature_names_out(self.config.categorical_cols)
            self.all_final_feature_names = self.numerical_feature_names + list(encoded_cat_names)
        elif self.config.encoding_type == "category":
            self.category_maps = {col: X[col].unique().tolist() for col in self.config.categorical_cols}
            self.all_final_feature_names = self.numerical_feature_names + self.config.categorical_cols
        else:
            raise ValueError(f"Unsupported encoding_type: {self.config.encoding_type}")
        
        # Transform the full training data using the newly fitted encoder/maps
        X_encoded  = transform_categorical_features(
            df=X,
            categorical_cols=self.config.categorical_cols,
            encoding_type=self.config.encoding_type,
            one_hot_encoder=self.one_hot_encoder,
            category_maps=self.category_maps,
            numerical_cols=self.numerical_feature_names,
            all_final_cols=self.all_final_feature_names
        )

        # Assign the final feature names to self.feature_names for model saving/loading consistency
        self.feature_names = self.all_final_feature_names

        # Create and train final model
        base_model = LogisticRegression(**self.config.hyperparams)
        #base_model = self._get_lgbm_model()
        base_model.fit(X_encoded, y)

        # Store base model to extract feature importance
        self.model = base_model

        self.calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method=self.config.calibration_method
        )

        self.calibrated_model.fit(X_encoded, y)
        self.is_trained = True

        logger.info("Final model training completed")

    def _get_transformed_data(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Helper method to transform raw input data using the stored encoder/category maps"""
        if self.all_final_feature_names is None or self.numerical_feature_names is None:
            raise ValueError("Model not fully trained or feature names not set. Call train_final_model first.")
        
        # Pass the correct encoder/map based on ecoding_type
        return transform_categorical_features(
            df=X_raw,
            categorical_cols=self.config.categorical_cols,
            encoding_type=self.config.encoding_type,
            one_hot_encoder=self.one_hot_encoder, # Will be None if encoding_type is 'category'
            category_maps=self.category_maps, # Will be None if encoding_type is 'one-hot'
            numerical_cols=self.numerical_feature_names,
            all_final_cols=self.all_final_feature_names
        )
    
    def plot_calibration_curve(self, X: pd.DataFrame, y: pd.Series, save_path: Optional[str] = None) -> None:
        """Plot calibration curve for the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting calibration curve")
        
        # Use the helper method to transform the data for plotting
        X_encoded = self._get_transformed_data(X)
        #X_encoded = X.copy()

        # Get probabilities
        y_pred_proba = self.calibrated_model.predict_proba(X_encoded)[:, 1]

        # Creat calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, y_pred_proba, n_bins=10, strategy='quantile'
        )

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14,6))

        # Calibration Curve
        axes[0].plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated CatBoost")
        axes[0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        axes[0].set_xlabel("Mean Predicted Probability")
        axes[0].set_ylabel("Fraction of Positives")
        axes[0].set_title("Calibration Curve | strategy = 'quantile'")
        axes[0].legend()
        axes[0].grid(True)
        
        # Histogram of probabilities
        axes[1].hist(y_pred_proba, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel("Predicted Probability")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Predicted Probabilities")
        axes[1].grid(True)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration of Predicted Probabilities")
            mlflow.log_artifact(save_path)
        
        plt.close(fig)

    def plot_feature_importance(self, save_path: Optional[str] = None, top_n: int=20) -> None:
        """Plot feature importance for base CatBoost model"""
        if not self.is_trained or self.model is None:
            raise ValueError("Base model not trained. Call train_final_model first.")
        
        if not hasattr(self.model, 'coef_'):
            logger.warning("Base model does not have coef_ attribute. Cannot plot importance.")
            return 
        
        importances = self.model.coef_[0]
        #feature_names = self.feature_names
        feature_names = self.model.feature_names_in_
        print("Importances...")
        print(len(importances))
        print("Features...")
        print(len(feature_names))

        if feature_names is None or len(feature_names) != len(importances):
            logger.warning("Feature names not available or mismatch with importances. Cannot plot importance.")
            return 
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        plt.figure(figsize=(10,8))
        sns.barplot(x='importance',y='feature', data=feature_importance_df.head(top_n))
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
            mlflow.log_artifact(save_path)
        plt.close()

    def save_model(self, file_path: str) -> None: 
        """Save the trained model and configuration"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.calibrated_model,
            'config': self.config,
            'results': self.results,
            'feature_names': self.feature_names,
            'one_hot_encoder': self.one_hot_encoder,
            'category_maps': self.category_maps,
            'numerical_feature_names': self.numerical_feature_names,
            'all_final_feature_names': self.all_final_feature_names,
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
        trainer.one_hot_encoder = model_data['one_hot_encoder']
        trainer.category_maps = model_data['category_maps']
        trainer.numerical_feature_names = model_data['numerical_feature_names']
        trainer.all_final_feature_names = model_data['all_final_feature_names']
        trainer.is_trained = True

        logger.info(f"Model loaded from {file_path}")

        return trainer
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model"""
        if not self.is_trained: 
            raise ValueError("Model must be trained before making predictions")
        
        X_encoded = self._get_transformed_data(X)
        #X_encoded = X.copy()

        return self.calibrated_model.predict(X_encoded)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model"""
        if not self.is_trained: 
            raise ValueError("Model must be trained before making predictions")
        
        X_encoded = self._get_transformed_data(X)
        #X_encoded = X.copy()

        return self.calibrated_model.predict_proba(X_encoded)
    
if __name__=='__main__':

    # Start MLflow run
    mlflow.set_experiment("MLB_Betting_Model_Training")
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        # Load data
        input_model_data = "data/processed/model_data.csv"
        df_model = load_model_data(input_model_data)
        df_model  = df_model[df_model['game_date']<datetime.today()]
        target = 'home_win'
        holdout_start_date = '2025-07-24'
        # Data splitting
        df_train_init, df_holdout, _ = data_split(
            df=df_model,
            holdout_start_date=holdout_start_date,
            group_col='game_date'
        )

        # Remove columns
        cols_to_remove = [
            "game_id", "game_date", "game_date_time", "home_team_id",
            "away_team_id", "home_score", "away_score", "state", target
        ]


        # Prepare features (X) and target (y) for training and holdout
        X_train_init = df_train_init.drop(cols_to_remove, axis=1)
        #feature_subset = ['home_team_games_prev_7days', 'home_team_season_opener_flag','home_team_rest_days','away_team']
        #X_train_init = X_train_init[feature_subset]
        y_train_init = df_train_init[target]

        X_holdout = df_holdout.drop(cols_to_remove, axis=1)
        #X_holdout = X_holdout[feature_subset]
        y_holdout = df_holdout[target]

        # Define groups for GroupTSCV - do i not need groups from data_split() ? --> will groups for holdout be useful?
        groups = df_train_init['game_date']

        

        # Configuration
        config = ModelConfig(
            model_type="logistic",
            hyperparams={
                'penalty': 'l1',
                'C': .005,
                'solver': 'liblinear'
                # 'n_estimators': 500,        # Increased estimators for LGBM
                # 'learning_rate': 0.05,      # Learning rate for boosting
                # 'num_leaves': 31,           # Controls complexity of trees
                # 'max_depth': -1,            # No limit on depth, often works well with num_leaves
                # 'min_child_samples': 20,    # Minimum data in a leaf
                # 'subsample': 0.8,           # Fraction of samples to be randomly sampled
                # 'colsample_bytree': 0.8,    # Fraction of features to be randomly sampled per tree
                # 'random_state': 888,
                # 'n_jobs': -1,
                # 'reg_alpha': 0.1,           # L1 regularization
                # 'reg_lambda': 0.1          # L2 regularization
            },
            cv_params={
                'test_size': 30,
                'train_size': 360,
                #'n_splits': 3, # for testing purposes
                'gap_size': 3,
                'window_type': 'rolling'
            },
            #categorical_cols=['away_team'], 
            categorical_cols=['home_team','away_team','venue','game_type'],
            encoding_type='one-hot',
            calibration_method='isotonic',
            random_state=888,
            holdout_start_date=holdout_start_date
        )

        # Log ModelConfig parameters
        mlflow.log_param("model_type", config.model_type)
        mlflow.log_params(config.hyperparams)
        mlflow.log_params(config.cv_params)
        mlflow.log_param("categorical_cols", config.categorical_cols)
        mlflow.log_param("encoding_type", config.encoding_type)
        mlflow.log_param("calibration_method", config.calibration_method)
        mlflow.log_param("random_state", config.random_state)
        mlflow.log_param("holdout_start_date", config.holdout_start_date)

        # Train model
        trainer = MLBModelTrainer(config)

        # Cross-validation
        # results = trainer.train_with_cross_validation(X_train_init, y_train_init, groups)

        # mlflow.log_metric("cv_mean_accuracy", results.mean_accuracy)
        # mlflow.log_metric("cv_mean_log_loss", results.mean_log_loss)
        # mlflow.log_metric("cv_mean_brier_score", results.mean_brier_score)
        # mlflow.log_metric("cv_mean_auc", results.mean_auc)
        # mlflow.log_metric("cv_mean_calibration_score", results.mean_calibration_score)

        # Train final model
        trainer.train_final_model(X_train_init, y_train_init)

        # Plot calibration curve
        trainer.plot_calibration_curve(X_train_init, y_train_init, save_path='src/figures/calibration_curve_train_logistic.png')
        # trainer.plot_calibration_curve(X_holdout, y_holdout, save_path='src/figures/calibration_curve_holdout_logistic.png')

        # Plot feature importances
        trainer.plot_feature_importance(save_path='src/figures/feature_importances_logistic_all_data.png', top_n = 20)

        # Save model
        trainer.save_model('src/saved_models/trained_mlb_model_logistic_all_data.pkl')

        # Input example for MLflow
        # sample_input_raw = X_train_init.head(1).copy()
        # sample_input_transformed = trainer._get_transformed_data(sample_input_raw)
        # input_schema_cols = []
        # for col in sample_input_transformed.columns:
        #     if col in config.categorical_cols:
        #         input_schema_cols.append(ColSpec("string", col))
        #     elif pd.api.types.is_integer_dtype(sample_input_transformed[col]):
        #         input_schema_cols.append(ColSpec("long", col))
        #     else:
        #         input_schema_cols.append(ColSpec("double", col))
        # input_schema = Schema(input_schema_cols)
        # output_schema = Schema([ColSpec("double", "prediction_probability")])
        # signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Log the trained CalibratedClassifierCV model
        mlflow.sklearn.log_model(
            sk_model=trainer.calibrated_model,
            name="mlb_model_logistic_all_data",
            registered_model_name="MLB_Calibrated_Logistic_Model",
            #input_example=sample_input_transformed,
            #signature=signature
        )

        # Evaluate on holdout set
        # holdout_predictions = trainer.predict(X_holdout)
        # holdout_probabilities = trainer.predict_proba(X_holdout)[:, 1]

        # holdout_accuracy = accuracy_score(y_holdout, holdout_predictions)
        # holdout_log_loss = log_loss(y_holdout, holdout_probabilities)
        # holdout_brier = brier_score_loss(y_holdout, holdout_probabilities)
        # holdout_auc = roc_auc_score(y_holdout, holdout_probabilities)

        # logger.info("\n" + "="*50)
        # logger.info("HOLDOUT SET EVALUTION")
        # logger.info("="*50)
        # logger.info(f"Holdout Accuracy: {holdout_accuracy: .4f}")
        # logger.info(f"Holdout Log Loss: {holdout_log_loss: .4f}")
        # logger.info(f"Holdout Brier Score: {holdout_brier: .4f}")
        # logger.info(f"Holdout AUC: {holdout_auc: .4f}")

        # mlflow.log_metric("holdout_accuracy", holdout_accuracy)
        # mlflow.log_metric("holdout_log_loss", holdout_log_loss)
        # mlflow.log_metric("holdout_brier_score", holdout_brier)
        # mlflow.log_metric("holdout_auc", holdout_auc)
    
    logger.info("MLflow run finished.")