# -*- coding: utf-8 -*-
"""
Training Script for Irrigation Need Prediction
Baseline: LightGBM with stratified k-fold cross-validation
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import yaml
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class IrrigationNeedModel:
    """Train and evaluate LightGBM model for irrigation need prediction"""

    def __init__(self, config):
        self.config = config
        self.model_params = config.get("model_params", {})
        self.feature_params = config.get("feature_params", {})
        self.cv_params = config.get("cv_params", {})
        self.random_state = config.get("random_state", 42)

        np.random.seed(self.random_state)

    def load_data(self, train_path, test_path=None):
        """Load and preprocess data"""
        print(f"Loading train data from {train_path}...")
        self.train_df = pd.read_csv(train_path)

        if test_path:
            print(f"Loading test data from {test_path}...")
            self.test_df = pd.read_csv(test_path)
        else:
            self.test_df = None

        print(f"Train shape: {self.train_df.shape}")
        if self.test_df is not None:
            print(f"Test shape: {self.test_df.shape}")

        # Identify target and features
        self.target_col = "Irrigation_Need"
        self.id_col = "id"

        # Check for null values
        print(f"\nNull values in train:\n{self.train_df.isnull().sum()}")

        return self.train_df, self.test_df

    def preprocess_data(self, df, is_train=True):
        """Preprocess features and handle encoding"""
        df = df.copy()

        # Separate id, target, and features
        if self.id_col in df.columns:
            ids = df[self.id_col].values
            df = df.drop(columns=[self.id_col])
        else:
            ids = None

        if is_train and self.target_col in df.columns:
            target = df[self.target_col].values
            df = df.drop(columns=[self.target_col])
        else:
            target = None

        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        print(f"\nCategorical features: {categorical_cols}")
        print(f"Numerical features: {numerical_cols}")

        # Encode categorical variables
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            if is_train:
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
            else:
                # For test set, use previously fitted encoder
                if col in self.le_dict:
                    df[col] = self.le_dict[col].transform(df[col].astype(str))

        if is_train:
            self.le_dict = le_dict
            self.categorical_cols = categorical_cols
            self.numerical_cols = numerical_cols

        return df.values, target, ids

    def train_and_evaluate(self):
        """Train with k-fold cross-validation"""
        X, y, _ = self.preprocess_data(self.train_df, is_train=True)

        n_splits = self.cv_params.get("n_splits", 5)
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.cv_params.get("random_state", 42),
        )

        # Store OOF predictions and models
        self.oof_predictions = np.zeros(len(y))
        self.models = []
        self.cv_scores = []

        print(f"\n{'='*60}")
        print(f"Starting {n_splits}-Fold Cross-Validation")
        print(f"{'='*60}\n")

        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'='*40} FOLD {fold + 1}/{n_splits} {'='*40}")

            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            print(f"Train size: {len(X_train)}, Valid size: {len(X_valid)}")
            print(
                f"Class distribution - Train: {np.mean(y_train):.3f}, Valid: {np.mean(y_valid):.3f}"
            )

            # Create LightGBM dataset
            train_data = lgb.Dataset(
                X_train,
                y_train,
                feature_names=[f"f_{i}" for i in range(X_train.shape[1])],
                free_raw_data=False,
            )
            valid_data = lgb.Dataset(
                X_valid, y_valid, reference=train_data, free_raw_data=False
            )

            # Train model
            model = lgb.train(
                self.model_params,
                train_data,
                num_boost_round=5000,
                valid_sets=[valid_data],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
            )

            # Predictions on validation fold
            oof_preds = model.predict(X_valid, num_iteration=model.best_iteration)
            self.oof_predictions[valid_idx] = oof_preds

            # Evaluate
            auc = roc_auc_score(y_valid, oof_preds)
            f1 = f1_score(y_valid, (oof_preds > 0.5).astype(int))
            precision = precision_score(y_valid, (oof_preds > 0.5).astype(int))
            recall = recall_score(y_valid, (oof_preds > 0.5).astype(int))

            fold_result = {
                "fold": fold,
                "auc": auc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
            self.cv_scores.append(fold_result)

            print(f"\nFold {fold + 1} Results:")
            print(f"  AUC:       {auc:.5f}")
            print(f"  F1:        {f1:.5f}")
            print(f"  Precision: {precision:.5f}")
            print(f"  Recall:    {recall:.5f}")

            self.models.append(model)

        # Overall CV scores
        overall_auc = roc_auc_score(y, self.oof_predictions)
        overall_f1 = f1_score(y, (self.oof_predictions > 0.5).astype(int))
        overall_precision = precision_score(y, (self.oof_predictions > 0.5).astype(int))
        overall_recall = recall_score(y, (self.oof_predictions > 0.5).astype(int))

        print(f"\n{'='*60}")
        print(f"Overall CV Results:")
        print(f"  AUC:       {overall_auc:.5f}")
        print(f"  F1:        {overall_f1:.5f}")
        print(f"  Precision: {overall_precision:.5f}")
        print(f"  Recall:    {overall_recall:.5f}")
        print(f"{'='*60}")

        self.overall_results = {
            "auc": overall_auc,
            "f1": overall_f1,
            "precision": overall_precision,
            "recall": overall_recall,
            "fold_results": self.cv_scores,
        }

        return self.overall_results

    def predict_test(self):
        """Generate test predictions"""
        if self.test_df is None:
            print("No test data available")
            return None

        X_test, _, test_ids = self.preprocess_data(self.test_df, is_train=False)

        # Average predictions from all folds
        test_predictions = np.zeros(len(X_test))
        for model in self.models:
            test_predictions += model.predict(
                X_test, num_iteration=model.best_iteration
            )
        test_predictions /= len(self.models)

        return test_predictions, test_ids

    def save_results(self, output_dir):
        """Save OOF predictions and results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save OOF predictions
        oof_df = pd.DataFrame(
            {
                "id": range(len(self.oof_predictions)),
                "Irrigation_Need_pred": self.oof_predictions,
            }
        )
        oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
        print(f"Saved OOF predictions to {output_dir / 'oof_predictions.csv'}")

        # Save results JSON
        results_json = {
            "cv_results": self.overall_results,
            "model_params": self.model_params,
            "config": self.config,
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"Saved results to {output_dir / 'results.json'}")

        # Save config
        if "config_path" in self.config:
            shutil.copy(self.config["config_path"], output_dir / "config.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM for Irrigation Need Prediction"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--train_data", type=str, default="data/train.csv", help="Path to training data"
    )
    parser.add_argument(
        "--test_data", type=str, default="data/test.csv", help="Path to test data"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["config_path"] = args.config

    print(f"\nConfig loaded from {args.config}:")
    print(yaml.dump(config, default_flow_style=False))

    # Initialize model
    model = IrrigationNeedModel(config)

    # Load data
    model.load_data(args.train_data, args.test_data)

    # Train and validate
    cv_results = model.train_and_evaluate()

    # Predict on test
    test_preds, test_ids = model.predict_test()

    # Save results
    exp_name = config.get("exp_name", "EXP001")
    child_exp_name = config.get("child_exp_name", "child-exp000")
    output_dir = f"EXP/{exp_name}/outputs/{child_exp_name}"

    model.save_results(output_dir)

    print(f"\n✓ Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
