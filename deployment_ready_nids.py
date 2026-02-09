"""
SentinelNet - Production Deployment System
Real-time Network Intrusion Detection with Alert Generation

This is a deployment-ready script that loads trained models and performs
real-time predictions on network traffic data, generating alerts for detected intrusions.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Import configuration
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'deployment.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DeploymentNIDS:
    """Production-ready Network Intrusion Detection System"""
    
    def __init__(self):
        """Initialize the deployment system"""
        self.model = None
        self.scaler = None
        self.pca = None
        self.model_name = None
        logger.info("Deployment NIDS initialized")
    
    def load_trained_model(self, model_name='Random Forest'):
        """Load pre-trained model and preprocessors"""
        logger.info(f"Loading trained model: {model_name}")
        
        try:
            # Load model
            model_file = os.path.join(
                config.MODEL_DIR,
                f'{model_name.replace(" ", "_").lower()}.pkl'
            )
            
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"Model not found: {model_file}\n"
                    "Please train the models first by running: python sentinelnet_main.py"
                )
            
            self.model = joblib.load(model_file)
            self.model_name = model_name
            logger.info(f"✓ Loaded model: {model_name}")
            
            # Load scaler
            scaler_file = os.path.join(config.MODEL_DIR, 'scaler.pkl')
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
                logger.info("✓ Loaded feature scaler")
            
            # Load PCA (if using SVM)
            if model_name == 'SVM':
                pca_file = os.path.join(config.MODEL_DIR, 'pca.pkl')
                if os.path.exists(pca_file):
                    self.pca = joblib.load(pca_file)
                    logger.info("✓ Loaded PCA transformer")
            
            logger.info("✓ Model and preprocessors loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            return False
    
    def preprocess_traffic(self, traffic_data):
        """Preprocess network traffic data for prediction"""
        try:
            # Ensure data is a DataFrame
            if not isinstance(traffic_data, pd.DataFrame):
                traffic_data = pd.DataFrame(traffic_data)
            
            # Apply scaling if scaler is available
            if self.scaler:
                traffic_scaled = self.scaler.transform(traffic_data)
                traffic_data = pd.DataFrame(traffic_scaled, columns=traffic_data.columns)
            
            # Apply PCA if available (for SVM)
            if self.pca and self.model_name == 'SVM':
                traffic_data = self.pca.transform(traffic_data)
            
            return traffic_data
            
        except Exception as e:
            logger.error(f"✗ Preprocessing failed: {e}")
            raise
    
    def predict_intrusion(self, traffic_data):
        """Predict whether traffic contains intrusions"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_trained_model() first.")
        
        try:
            # Preprocess
            processed_data = self.preprocess_traffic(traffic_data)
            
            # Predict
            predictions = self.model.predict(processed_data)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"✗ Prediction failed: {e}")
            raise
    
    def generate_alert(self, prediction, sample_id, confidence=None):
        """Generate an alert for detected intrusion"""
        if prediction == 'Normal':
            return None
        
        severity = config.ALERT_SEVERITY.get(prediction, 0)
        
        alert = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'sample_id': sample_id,
            'attack_type': prediction,
            'severity': severity,
            'severity_level': self._get_severity_level(severity),
            'confidence': f"{confidence:.2%}" if confidence else "N/A",
            'action': self._get_recommended_action(severity)
        }
        
        return alert
    
    def _get_severity_level(self, severity):
        """Get human-readable severity level"""
        levels = {0: 'INFO', 1: 'LOW', 2: 'MEDIUM', 3: 'HIGH', 4: 'CRITICAL'}
        return levels.get(severity, 'UNKNOWN')
    
    def _get_recommended_action(self, severity):
        """Get recommended action based on severity"""
        actions = {
            0: 'Monitor',
            1: 'Log and monitor',
            2: 'Investigate and log',
            3: 'Block and investigate',
            4: 'Immediate block and alert admin'
        }
        return actions.get(severity, 'Review')
    
    def log_alert(self, alert):
        """Log alert to file"""
        if alert is None:
            return
        
        try:
            alert_file = os.path.join(config.LOGS_DIR, 'real_time_alerts.csv')
            
            # Create DataFrame
            alert_df = pd.DataFrame([alert])
            
            # Append to file
            if os.path.exists(alert_file):
                alert_df.to_csv(alert_file, mode='a', header=False, index=False)
            else:
                alert_df.to_csv(alert_file, mode='w', header=True, index=False)
            
            # Log to console with color coding
            severity_level = alert['severity_level']
            if severity_level in ['HIGH', 'CRITICAL']:
                prefix = "🚨 ALERT"
            elif severity_level == 'MEDIUM':
                prefix = "⚠️  WARNING"
            else:
                prefix = "ℹ️  INFO"
            
            logger.warning(
                f"{prefix} [{alert['timestamp']}] "
                f"{alert['attack_type']} detected (Severity: {severity_level}) "
                f"- {alert['action']}"
            )
            
        except Exception as e:
            logger.error(f"✗ Failed to log alert: {e}")
    
    def process_batch(self, traffic_data, batch_name="batch"):
        """Process a batch of network traffic and generate alerts"""
        logger.info(f"\nProcessing {len(traffic_data)} traffic samples...")
        
        try:
            # Predict
            predictions, probabilities = self.predict_intrusion(traffic_data)
            
            # Generate and log alerts
            alert_count = 0
            for i, prediction in enumerate(predictions):
                confidence = None
                if probabilities is not None:
                    # Get confidence for predicted class
                    pred_class_idx = list(self.model.classes_).index(prediction)
                    confidence = probabilities[i][pred_class_idx]
                
                alert = self.generate_alert(prediction, f"{batch_name}_{i}", confidence)
                
                if alert:
                    self.log_alert(alert)
                    alert_count += 1
            
            # Summary
            normal_count = (predictions == 'Normal').sum()
            logger.info(f"\n✓ Processing complete:")
            logger.info(f"  Normal traffic: {normal_count}/{len(predictions)}")
            logger.info(f"  Intrusions detected: {alert_count}/{len(predictions)}")
            
            return predictions, alert_count
            
        except Exception as e:
            logger.error(f"✗ Batch processing failed: {e}")
            raise
    
    def monitor_traffic_file(self, file_path):
        """Monitor and process traffic from a CSV file"""
        logger.info(f"Monitoring traffic file: {file_path}")
        
        try:
            # Load traffic data
            traffic_data = pd.read_csv(file_path, names=config.COLUMN_NAMES, header=None)
            
            # Drop label and difficulty if present
            if 'label' in traffic_data.columns:
                traffic_data = traffic_data.drop(['label', 'difficulty'], axis=1)
            
            logger.info(f"✓ Loaded {len(traffic_data)} traffic samples")
            
            # Process in batches
            batch_size = 100
            total_alerts = 0
            
            for i in range(0, len(traffic_data), batch_size):
                batch = traffic_data.iloc[i:i+batch_size]
                batch_name = f"batch_{i//batch_size}"
                
                _, alert_count = self.process_batch(batch, batch_name)
                total_alerts += alert_count
            
            logger.info(f"\n✓ Monitoring complete. Total alerts: {total_alerts}")
            
        except Exception as e:
            logger.error(f"✗ Traffic monitoring failed: {e}")
            raise


def demo_deployment():
    """Demonstrate the deployment system"""
    print("\n" + "=" * 80)
    print("  SENTINELNET - PRODUCTION DEPLOYMENT DEMO")
    print("  Real-time Network Intrusion Detection")
    print("=" * 80 + "\n")
    
    # Initialize deployment system
    nids = DeploymentNIDS()
    
    # Load trained model
    if not nids.load_trained_model('Random Forest'):
        print("\n✗ Failed to load model. Please train the models first:")
        print("  python sentinelnet_main.py")
        return
    
    # Check if test data exists
    if os.path.exists(config.TEST_DATA_PATH):
        print("\n📊 Running demo with test dataset...")
        nids.monitor_traffic_file(config.TEST_DATA_PATH)
    else:
        print("\n⚠️  Test dataset not found.")
        print("Please run the main pipeline first: python run_project.py")
    
    print("\n" + "=" * 80)
    print("  Demo complete! Check logs/real_time_alerts.csv for results.")
    print("=" * 80 + "\n")


def main():
    """Main entry point for deployment system"""
    demo_deployment()


if __name__ == "__main__":
    main()