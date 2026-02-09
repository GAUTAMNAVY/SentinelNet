"""
SentinelNet - AI-Powered Network Intrusion Detection System
Main Pipeline Implementation - All 8 Modules

This script implements the complete NIDS pipeline covering:
- Module 1: Dataset Acquisition and Exploration
- Module 2: Data Cleaning and Preprocessing
- Module 3: Feature Engineering and Selection
- Module 4: Model Building and Training (Supervised)
- Module 5: Anomaly Detection (Unsupervised)
- Module 6: Model Evaluation and Tuning
- Module 7: Alert Generation and Logging
- Module 8: Documentation and Presentation
"""

import os
import sys
import warnings
import logging
from datetime import datetime

# Data processing
import pandas as pd
import numpy as np

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import requests
import joblib

# Import configuration
import config

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'sentinelnet.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SentinelNetNIDS:
    """Main class for SentinelNet Network Intrusion Detection System"""
    
    def __init__(self):
        """Initialize the NIDS pipeline"""
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.label_encoder = None
        self.pca = None
        self.results = {}
        
        logger.info("SentinelNet NIDS Initialized")
    
    # ========================================================================
    # MODULE 1: Dataset Acquisition and Exploration
    # ========================================================================
    
    def download_dataset(self):
        """Download NSL-KDD dataset if not already present"""
        logger.info("=" * 80)
        logger.info("MODULE 1: Dataset Acquisition and Exploration")
        logger.info("=" * 80)
        
        for dataset_type, url in config.DATASET_URLS.items():
            file_path = config.TRAIN_DATA_PATH if dataset_type == 'train' else config.TEST_DATA_PATH
            
            if os.path.exists(file_path):
                logger.info(f"✓ {dataset_type.capitalize()} dataset already exists: {file_path}")
            else:
                logger.info(f"Downloading {dataset_type} dataset from {url}...")
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"✓ Successfully downloaded {dataset_type} dataset")
                except Exception as e:
                    logger.error(f"✗ Error downloading {dataset_type} dataset: {e}")
                    raise
    
    def load_and_explore_data(self):
        """Load and perform initial exploration of the dataset"""
        logger.info("\nLoading datasets...")
        
        # Load train and test data
        self.train_data = pd.read_csv(
            config.TRAIN_DATA_PATH,
            names=config.COLUMN_NAMES,
            header=None
        )
        
        self.test_data = pd.read_csv(
            config.TEST_DATA_PATH,
            names=config.COLUMN_NAMES,
            header=None
        )
        
        logger.info(f"✓ Train dataset shape: {self.train_data.shape}")
        logger.info(f"✓ Test dataset shape: {self.test_data.shape}")
        
        # Basic statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"  Total training samples: {len(self.train_data)}")
        logger.info(f"  Total test samples: {len(self.test_data)}")
        logger.info(f"  Total features: {len(config.COLUMN_NAMES) - 2}")  # Excluding label and difficulty
        
        # Attack type distribution
        logger.info("\nAttack Type Distribution (Training Set):")
        attack_counts = self.train_data['label'].value_counts()
        for attack, count in attack_counts.head(10).items():
            logger.info(f"  {attack}: {count}")
        
        # Data types
        logger.info(f"\nData Types:")
        logger.info(f"  Numerical features: {self.train_data.select_dtypes(include=[np.number]).shape[1]}")
        logger.info(f"  Categorical features: {self.train_data.select_dtypes(include=['object']).shape[1]}")
        
        # Check for missing values
        missing_train = self.train_data.isnull().sum().sum()
        missing_test = self.test_data.isnull().sum().sum()
        logger.info(f"\nMissing Values:")
        logger.info(f"  Training set: {missing_train}")
        logger.info(f"  Test set: {missing_test}")
        
        return self.train_data, self.test_data
    
    # ========================================================================
    # MODULE 2: Data Cleaning and Preprocessing
    # ========================================================================
    
    def preprocess_data(self):
        """Clean and preprocess the dataset"""
        logger.info("\n" + "=" * 80)
        logger.info("MODULE 2: Data Cleaning and Preprocessing")
        logger.info("=" * 80)
        
        # Create a copy to avoid modifying original data
        train_processed = self.train_data.copy()
        test_processed = self.test_data.copy()
        
        # Map attack types to categories
        logger.info("\nMapping attack types to categories...")
        train_processed['attack_category'] = train_processed['label'].apply(self._categorize_attack)
        test_processed['attack_category'] = test_processed['label'].apply(self._categorize_attack)
        
        # Drop difficulty column (not needed for training)
        train_processed = train_processed.drop('difficulty', axis=1)
        test_processed = test_processed.drop('difficulty', axis=1)
        
        # Encode categorical features
        logger.info("Encoding categorical features...")
        categorical_columns = ['protocol_type', 'service', 'flag']
        
        for col in categorical_columns:
            le = LabelEncoder()
            # Fit on combined data to ensure consistent encoding
            combined = pd.concat([train_processed[col], test_processed[col]])
            le.fit(combined)
            train_processed[col] = le.transform(train_processed[col])
            test_processed[col] = le.transform(test_processed[col])
        
        logger.info(f"✓ Encoded {len(categorical_columns)} categorical features")
        
        # Separate features and labels
        X_train = train_processed.drop(['label', 'attack_category'], axis=1)
        y_train = train_processed['attack_category']
        
        X_test = test_processed.drop(['label', 'attack_category'], axis=1)
        y_test = test_processed['attack_category']
        
        # Normalize numerical features
        logger.info("Normalizing numerical features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        self.X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        self.X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        
        logger.info(f"✓ Preprocessed training set shape: {self.X_train.shape}")
        logger.info(f"✓ Preprocessed test set shape: {self.X_test.shape}")
        
        # Class distribution
        logger.info("\nClass Distribution:")
        for category in self.y_train.unique():
            train_count = (self.y_train == category).sum()
            test_count = (self.y_test == category).sum()
            logger.info(f"  {category}: Train={train_count}, Test={test_count}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _categorize_attack(self, label):
        """Categorize attack types"""
        label = label.strip()
        
        if label == 'normal':
            return 'Normal'
        
        for category, attacks in config.ATTACK_CATEGORIES.items():
            if category == 'normal':
                continue
            if label in attacks:
                return category
        
        # Unknown attack types categorized as 'Other'
        return 'Other'
    
    # ========================================================================
    # MODULE 3: Feature Engineering and Selection
    # ========================================================================
    
    def feature_engineering(self):
        """Perform feature engineering and selection"""
        logger.info("\n" + "=" * 80)
        logger.info("MODULE 3: Feature Engineering and Selection")
        logger.info("=" * 80)
        
        # Feature importance using Random Forest
        logger.info("\nAnalyzing feature importance...")
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE, n_jobs=-1)
        rf_temp.fit(self.X_train, self.y_train)
        
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save feature importance plot
        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance['feature'].head(20), feature_importance['importance'].head(20))
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'feature_importance.png'), dpi=config.DPI)
        plt.close()
        logger.info(f"✓ Saved feature importance plot")
        
        # Correlation analysis
        logger.info("\nPerforming correlation analysis...")
        correlation_matrix = self.X_train.corr().abs()
        
        # Find highly correlated features
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > config.FEATURE_CORRELATION_THRESHOLD)
        ]
        
        logger.info(f"✓ Found {len(high_corr_features)} highly correlated features")
        
        # PCA for dimensionality reduction
        logger.info(f"\nApplying PCA (components={config.PCA_COMPONENTS})...")
        self.pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_STATE)
        X_train_pca = self.pca.fit_transform(self.X_train)
        X_test_pca = self.pca.transform(self.X_test)
        
        explained_variance = sum(self.pca.explained_variance_ratio_)
        logger.info(f"✓ PCA explained variance: {explained_variance:.2%}")
        
        # Store PCA-transformed data
        self.X_train_pca = X_train_pca
        self.X_test_pca = X_test_pca
        
        return feature_importance
    
    # ========================================================================
    # MODULE 4: Model Building and Training (Supervised Learning)
    # ========================================================================
    
    def train_supervised_models(self):
        """Train supervised learning models"""
        logger.info("\n" + "=" * 80)
        logger.info("MODULE 4: Model Building and Training (Supervised Learning)")
        logger.info("=" * 80)
        
        # Random Forest
        logger.info("\n[1/3] Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(**config.MODEL_PARAMS['random_forest'])
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        logger.info("✓ Random Forest training complete")
        
        # Logistic Regression
        logger.info("\n[2/3] Training Logistic Regression...")
        lr_model = LogisticRegression(**config.MODEL_PARAMS['logistic_regression'])
        lr_model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr_model
        logger.info("✓ Logistic Regression training complete")
        
        # SVM (using PCA-transformed data for efficiency)
        logger.info("\n[3/3] Training SVM (on PCA features)...")
        svm_model = SVC(**config.MODEL_PARAMS['svm'], probability=True)
        svm_model.fit(self.X_train_pca, self.y_train)
        self.models['SVM'] = svm_model
        logger.info("✓ SVM training complete")
        
        # Quick evaluation
        logger.info("\nQuick Model Evaluation:")
        for name, model in self.models.items():
            if name == 'SVM':
                y_pred = model.predict(self.X_test_pca)
            else:
                y_pred = model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            logger.info(f"  {name}: {accuracy:.4f}")
        
        return self.models
    
    # ========================================================================
    # MODULE 5: Anomaly Detection (Unsupervised Learning)
    # ========================================================================
    
    def anomaly_detection(self):
        """Perform anomaly detection using unsupervised learning"""
        logger.info("\n" + "=" * 80)
        logger.info("MODULE 5: Anomaly Detection (Unsupervised Learning)")
        logger.info("=" * 80)
        
        # Isolation Forest
        logger.info("\n[1/2] Training Isolation Forest...")
        iso_forest = IsolationForest(**config.MODEL_PARAMS['isolation_forest'])
        iso_forest.fit(self.X_train)
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        train_anomalies = iso_forest.predict(self.X_train)
        test_anomalies = iso_forest.predict(self.X_test)
        
        train_anomaly_count = (train_anomalies == -1).sum()
        test_anomaly_count = (test_anomalies == -1).sum()
        
        logger.info(f"✓ Training anomalies detected: {train_anomaly_count} ({train_anomaly_count/len(train_anomalies):.2%})")
        logger.info(f"✓ Test anomalies detected: {test_anomaly_count} ({test_anomaly_count/len(test_anomalies):.2%})")
        
        self.models['Isolation Forest'] = iso_forest
        
        # K-Means Clustering
        logger.info("\n[2/2] Training K-Means Clustering...")
        kmeans = KMeans(**config.MODEL_PARAMS['kmeans'])
        kmeans.fit(self.X_train)
        
        train_clusters = kmeans.predict(self.X_train)
        test_clusters = kmeans.predict(self.X_test)
        
        logger.info(f"✓ K-Means clustering complete")
        logger.info(f"  Cluster distribution (test): {np.bincount(test_clusters)}")
        
        self.models['K-Means'] = kmeans
        
        return iso_forest, kmeans
    
    # ========================================================================
    # MODULE 6: Model Evaluation and Tuning
    # ========================================================================
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        logger.info("\n" + "=" * 80)
        logger.info("MODULE 6: Model Evaluation and Tuning")
        logger.info("=" * 80)
        
        evaluation_results = {}
        
        for name in ['Random Forest', 'Logistic Regression', 'SVM']:
            logger.info(f"\nEvaluating {name}...")
            model = self.models[name]
            
            # Predictions
            if name == 'SVM':
                y_pred = model.predict(self.X_test_pca)
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_test_pca)
                else:
                    y_pred_proba = None
            else:
                y_pred = model.predict(self.X_test)
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_test)
                else:
                    y_pred_proba = None
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(config.RESULTS_DIR, f'confusion_matrix_{name.replace(" ", "_").lower()}.png'), dpi=config.DPI)
            plt.close()
            
            logger.info(f"  ✓ Saved confusion matrix")
            
            # ROC Curve (for binary classification or One-vs-Rest)
            if y_pred_proba is not None and len(np.unique(self.y_test)) <= 5:
                self._plot_roc_curve(name, y_pred_proba)
        
        # Select best model
        best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['f1_score'])
        self.best_model = self.models[best_model_name]
        
        logger.info(f"\n✓ Best Model: {best_model_name} (F1-Score: {evaluation_results[best_model_name]['f1_score']:.4f})")
        
        self.results = evaluation_results
        return evaluation_results
    
    def _plot_roc_curve(self, model_name, y_pred_proba):
        """Plot ROC curve for multiclass classification"""
        try:
            from sklearn.preprocessing import label_binarize
            from itertools import cycle
            
            # Binarize labels
            classes = np.unique(self.y_test)
            y_test_bin = label_binarize(self.y_test, classes=classes)
            
            n_classes = y_test_bin.shape[1]
            
            # Compute ROC curve for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot
            plt.figure(figsize=(10, 8))
            colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
            
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(config.RESULTS_DIR, f'roc_curve_{model_name.replace(" ", "_").lower()}.png'), dpi=config.DPI)
            plt.close()
            
            logger.info(f"  ✓ Saved ROC curve")
        except Exception as e:
            logger.warning(f"  Could not generate ROC curve: {e}")
    
    # ========================================================================
    # MODULE 7: Alert Generation and Logging
    # ========================================================================
    
    def generate_alerts(self, sample_size=100):
        """Generate alerts for detected intrusions"""
        logger.info("\n" + "=" * 80)
        logger.info("MODULE 7: Alert Generation and Logging")
        logger.info("=" * 80)
        
        # Use best model for predictions
        logger.info(f"\nGenerating alerts using best model...")
        
        # Sample from test set
        sample_indices = np.random.choice(len(self.X_test), min(sample_size, len(self.X_test)), replace=False)
        X_sample = self.X_test.iloc[sample_indices]
        y_true_sample = self.y_test.iloc[sample_indices]
        
        # Predict
        best_model_name = [name for name, model in self.models.items() if model == self.best_model][0]
        
        if best_model_name == 'SVM':
            y_pred_sample = self.best_model.predict(self.pca.transform(X_sample))
        else:
            y_pred_sample = self.best_model.predict(X_sample)
        
        # Create alert log
        alert_data = []
        
        for i, (true_label, pred_label) in enumerate(zip(y_true_sample, y_pred_sample)):
            if pred_label != 'Normal':
                severity = config.ALERT_SEVERITY.get(pred_label, 0)
                alert_data.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'sample_id': sample_indices[i],
                    'predicted_attack': pred_label,
                    'true_label': true_label,
                    'severity': severity,
                    'status': 'ALERT' if pred_label == true_label else 'FALSE_POSITIVE'
                })
        
        # Save to CSV
        if alert_data:
            alert_df = pd.DataFrame(alert_data)
            alert_df.to_csv(config.ALERT_LOG_FILE, index=False)
            logger.info(f"✓ Generated {len(alert_data)} alerts")
            logger.info(f"✓ Alerts saved to: {config.ALERT_LOG_FILE}")
            
            # Display sample alerts
            logger.info("\nSample Alerts:")
            for idx, alert in enumerate(alert_data[:5]):
                logger.info(f"  [{alert['timestamp']}] {alert['predicted_attack']} - Severity: {alert['severity']} - {alert['status']}")
        else:
            logger.info("✓ No intrusions detected in sample")
        
        # Save prediction log
        prediction_df = pd.DataFrame({
            'sample_id': sample_indices,
            'true_label': y_true_sample,
            'predicted_label': y_pred_sample,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        prediction_df.to_csv(config.PREDICTION_LOG_FILE, index=False)
        logger.info(f"✓ Predictions saved to: {config.PREDICTION_LOG_FILE}")
        
        return alert_data
    
    # ========================================================================
    # MODULE 8: Documentation and Presentation
    # ========================================================================
    
    def generate_report(self):
        """Generate comprehensive project report"""
        logger.info("\n" + "=" * 80)
        logger.info("MODULE 8: Documentation and Presentation Preparation")
        logger.info("=" * 80)
        
        report = []
        report.append("=" * 80)
        report.append("SENTINELNET - AI-POWERED NETWORK INTRUSION DETECTION SYSTEM")
        report.append("Final Project Report")
        report.append("=" * 80)
        report.append("")
        
        # Dataset Information
        report.append("1. DATASET INFORMATION")
        report.append("-" * 80)
        report.append(f"   Training samples: {len(self.train_data)}")
        report.append(f"   Test samples: {len(self.test_data)}")
        report.append(f"   Total features: {self.X_train.shape[1]}")
        report.append(f"   Attack categories: {len(self.y_train.unique())}")
        report.append("")
        
        # Model Performance
        report.append("2. MODEL PERFORMANCE SUMMARY")
        report.append("-" * 80)
        
        for model_name, results in self.results.items():
            report.append(f"\n   {model_name}:")
            report.append(f"      Accuracy:  {results['accuracy']:.4f}")
            report.append(f"      Precision: {results['precision']:.4f}")
            report.append(f"      Recall:    {results['recall']:.4f}")
            report.append(f"      F1-Score:  {results['f1_score']:.4f}")
        
        report.append("")
        
        # Best Model
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
        report.append("3. BEST MODEL")
        report.append("-" * 80)
        report.append(f"   Selected Model: {best_model_name}")
        report.append(f"   F1-Score: {self.results[best_model_name]['f1_score']:.4f}")
        report.append("")
        
        # Outputs Generated
        report.append("4. OUTPUTS GENERATED")
        report.append("-" * 80)
        report.append(f"   - Feature importance plot: {os.path.join(config.RESULTS_DIR, 'feature_importance.png')}")
        report.append(f"   - Confusion matrices: {config.RESULTS_DIR}")
        report.append(f"   - ROC curves: {config.RESULTS_DIR}")
        report.append(f"   - Alert logs: {config.ALERT_LOG_FILE}")
        report.append(f"   - Prediction logs: {config.PREDICTION_LOG_FILE}")
        report.append("")
        
        # Conclusion
        report.append("5. CONCLUSION")
        report.append("-" * 80)
        report.append("   Successfully implemented an AI-powered Network Intrusion Detection System")
        report.append("   with supervised and unsupervised learning techniques. The system can")
        report.append("   accurately detect and classify various types of network attacks in real-time.")
        report.append("")
        report.append("=" * 80)
        
        # Print report
        report_text = "\n".join(report)
        logger.info("\n" + report_text)
        
        # Save report
        report_file = os.path.join(config.RESULTS_DIR, 'project_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"\n✓ Report saved to: {report_file}")
        
        return report_text
    
    # ========================================================================
    # Model Persistence
    # ========================================================================
    
    def save_models(self):
        """Save trained models to disk"""
        logger.info("\nSaving models...")
        
        # Save each model
        for name, model in self.models.items():
            model_file = os.path.join(config.MODEL_DIR, f'{name.replace(" ", "_").lower()}.pkl')
            joblib.dump(model, model_file)
            logger.info(f"  ✓ Saved {name} to {model_file}")
        
        # Save scaler and PCA
        joblib.dump(self.scaler, os.path.join(config.MODEL_DIR, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(config.MODEL_DIR, 'pca.pkl'))
        
        logger.info("✓ All models and preprocessors saved")
    
    def load_models(self):
        """Load trained models from disk"""
        logger.info("Loading models...")
        
        model_files = {
            'Random Forest': 'random_forest.pkl',
            'Logistic Regression': 'logistic_regression.pkl',
            'SVM': 'svm.pkl',
            'Isolation Forest': 'isolation_forest.pkl',
            'K-Means': 'k-means.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(config.MODEL_DIR, filename)
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                logger.info(f"  ✓ Loaded {name}")
        
        # Load scaler and PCA
        scaler_path = os.path.join(config.MODEL_DIR, 'scaler.pkl')
        pca_path = os.path.join(config.MODEL_DIR, 'pca.pkl')
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info("  ✓ Loaded scaler")
        
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
            logger.info("  ✓ Loaded PCA")
        
        logger.info("✓ Models loaded successfully")
    
    # ========================================================================
    # Main Execution Pipeline
    # ========================================================================
    
    def run_full_pipeline(self):
        """Execute the complete NIDS pipeline"""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING SENTINELNET NIDS PIPELINE")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Module 1
            self.download_dataset()
            self.load_and_explore_data()
            
            # Module 2
            self.preprocess_data()
            
            # Module 3
            self.feature_engineering()
            
            # Module 4
            self.train_supervised_models()
            
            # Module 5
            self.anomaly_detection()
            
            # Module 6
            self.evaluate_models()
            
            # Module 7
            self.generate_alerts()
            
            # Module 8
            self.generate_report()
            
            # Save models
            self.save_models()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "=" * 80)
            logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {duration:.2f} seconds")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}")
            raise


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("  SENTINELNET - AI-POWERED NETWORK INTRUSION DETECTION SYSTEM")
    print("=" * 80 + "\n")
    
    # Initialize and run pipeline
    nids = SentinelNetNIDS()
    nids.run_full_pipeline()
    
    print("\n" + "=" * 80)
    print("  All modules executed successfully!")
    print("  Check the 'results' and 'logs' directories for outputs.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()