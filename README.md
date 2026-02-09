# SentinelNet - AI-Powered Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)

An advanced AI-powered Network Intrusion Detection System (NIDS) that uses
machine learning to detect and classify cyber attacks in real-time.

## 📋 Overview

SentinelNet is a comprehensive NIDS solution that leverages both supervised and
unsupervised machine learning techniques to identify malicious network traffic
and cyber-attacks. The system is trained on the NSL-KDD dataset and achieves
high accuracy in detecting various types of network intrusions.

### Key Features

- ✅ **Multi-Model Approach**: Random Forest, SVM, and Logistic Regression
- ✅ **Anomaly Detection**: Isolation Forest and K-Means clustering
- ✅ **Real-time Alerts**: Automatic alert generation with severity levels
- ✅ **Comprehensive Logging**: Detailed logs and prediction tracking
- ✅ **Production Ready**: Deployment-ready inference system
- ✅ **Advanced Metrics**: Confusion matrices, ROC curves, and performance
  reports

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd c:\infosys
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

**Option 1: Complete Pipeline (Recommended for first run)**

```bash
python run_project.py
```

This executes all 8 modules in sequence and generates complete outputs.

**Option 2: Interactive Milestone Execution**

```bash
python run_milestones.py
```

Run specific milestones or all at once through an interactive menu.

**Option 3: Production Deployment Demo**

```bash
python deployment_ready_nids.py
```

Demonstrates real-time intrusion detection using trained models.

## 📊 Project Structure

```
c:\infosys/
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
├── sentinelnet_main.py           # Main pipeline (all 8 modules)
├── run_project.py                 # Quick execution script
├── run_milestones.py             # Interactive milestone runner
├── deployment_ready_nids.py      # Production deployment system
├── data/                          # Downloaded datasets
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── models/                        # Trained models
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── isolation_forest.pkl
│   ├── k-means.pkl
│   ├── scaler.pkl
│   └── pca.pkl
├── results/                       # Visualizations and reports
│   ├── feature_importance.png
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   └── project_report.txt
└── logs/                          # Alert and prediction logs
    ├── sentinelnet.log
    ├── intrusion_alerts.csv
    └── predictions.csv
```

## 📚 Module Breakdown

### Module 1: Dataset Acquisition and Exploration

- Downloads NSL-KDD dataset automatically
- Performs exploratory data analysis
- Generates basic statistics and attack distribution

### Module 2: Data Cleaning and Preprocessing

- Handles missing values and duplicates
- Encodes categorical features
- Normalizes numerical features
- Splits data for training and testing

### Module 3: Feature Engineering and Selection

- Feature importance analysis using Random Forest
- Correlation analysis for identifying redundant features
- PCA for dimensionality reduction

### Module 4: Model Building and Training (Supervised)

- **Random Forest**: Ensemble learning for robust predictions
- **SVM**: Support Vector Machine with RBF kernel
- **Logistic Regression**: Baseline linear classifier

### Module 5: Anomaly Detection (Unsupervised)

- **Isolation Forest**: Detects outliers in network traffic
- **K-Means Clustering**: Groups similar traffic patterns

### Module 6: Model Evaluation and Tuning

- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrices for all models
- ROC curves for performance visualization
- Model comparison and best model selection

### Module 7: Alert Generation and Logging

- Real-time prediction on test data
- Alert generation with severity levels:
  - **CRITICAL** (U2R attacks)
  - **HIGH** (R2L attacks)
  - **MEDIUM** (DoS attacks)
  - **LOW** (Probe attacks)
  - **INFO** (Normal traffic)
- CSV logging for audit trails

### Module 8: Documentation and Presentation

- Comprehensive project report
- Performance summaries
- Visualization outputs

## 🎯 Attack Types Detected

SentinelNet can detect and classify the following attack categories:

| Category   | Description           | Examples                             |
| ---------- | --------------------- | ------------------------------------ |
| **DoS**    | Denial of Service     | neptune, smurf, pod, teardrop        |
| **Probe**  | Surveillance/Scanning | portsweep, ipsweep, nmap, satan      |
| **R2L**    | Remote to Local       | ftp_write, guess_passwd, imap, phf   |
| **U2R**    | User to Root          | buffer_overflow, loadmodule, rootkit |
| **Normal** | Legitimate Traffic    | -                                    |

## 📈 Performance Metrics

After training, the system typically achieves:

- **Accuracy**: >90%
- **Precision**: >88%
- **Recall**: >90%
- **F1-Score**: >89%

_Results may vary based on dataset and configuration_

## 🔧 Configuration

Edit `config.py` to customize:

- Model hyperparameters
- Feature selection thresholds
- Alert severity levels
- File paths and directories
- Logging settings

## 📖 Usage Examples

### Running Specific Milestones

```python
from sentinelnet_main import SentinelNetNIDS

# Initialize
nids = SentinelNetNIDS()

# Run Module 1: Data Acquisition
nids.download_dataset()
nids.load_and_explore_data()

# Run Module 2: Preprocessing
nids.preprocess_data()

# Continue with other modules...
```

### Using Trained Models for Prediction

```python
from deployment_ready_nids import DeploymentNIDS
import pandas as pd

# Initialize deployment system
nids = DeploymentNIDS()
nids.load_trained_model('Random Forest')

# Load your traffic data
traffic_data = pd.read_csv('your_traffic_file.csv')

# Process and get alerts
predictions, alert_count = nids.process_batch(traffic_data)
```

## 📊 Outputs

### Visualizations

- **Feature Importance Plot**: Shows the most influential features
- **Confusion Matrices**: One for each model showing prediction accuracy
- **ROC Curves**: Model performance visualization

### Logs

- **intrusion_alerts.csv**: Detected intrusions with timestamps and severity
- **predictions.csv**: All predictions made by the system
- **sentinelnet.log**: Complete execution log

### Reports

- **project_report.txt**: Comprehensive summary of results

## 🛠️ Troubleshooting

**Issue**: Dataset download fails

- **Solution**: Check internet connection and firewall settings

**Issue**: Out of memory during training

- **Solution**: Reduce dataset size or adjust model parameters in `config.py`

**Issue**: Models not found for deployment

- **Solution**: Run `python run_project.py` first to train and save models

## 📝 Dataset Information

- **Name**: NSL-KDD Dataset
- **Source**: Canadian Institute for Cybersecurity
- **Training Samples**: ~125,000
- **Test Samples**: ~22,000
- **Features**: 41 features + 1 label
- **Classes**: 5 (Normal + 4 attack categories)

## 🎓 Academic Context

This project fulfills the requirements for an 8-week AI/ML internship covering:

- Machine Learning fundamentals
- Classification algorithms
- Anomaly detection
- Model evaluation techniques
- Real-world application deployment

## 🔐 Security Considerations

- This is a detection system, not a prevention system
- Alerts should be reviewed by security professionals
- False positives are possible and should be investigated
- Regular model retraining is recommended for evolving threats

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Improvements and suggestions are welcome! Areas for enhancement:

- Deep learning models (LSTM, CNN)
- Additional datasets (CICIDS2017, UNSW-NB15)
- Real-time packet capture integration
- Web-based dashboard

## 📧 Contact

For questions or issues, please open an issue in the project repository.

---

**Built with ❤️ using Python and Scikit-Learn**
