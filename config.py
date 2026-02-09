"""
SentinelNet - AI-Powered NIDS Configuration
Centralizes all project settings, paths, and parameters
"""

import os

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# NSL-KDD Dataset URLs
DATASET_URLS = {
    'train': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt',
    'test': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt'
}

# Column names for NSL-KDD dataset
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Attack type categories
ATTACK_CATEGORIES = {
    'normal': 'Normal',
    'DoS': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'worm'],
    'Probe': ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint'],
    'R2L': ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named'],
    'U2R': ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']
}

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# File paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'KDDTrain+.txt')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'KDDTest+.txt')

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Train-test split ratio
TEST_SIZE = 0.2

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'random_state': RANDOM_STATE
    },
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'isolation_forest': {
        'n_estimators': 100,
        'contamination': 0.1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'kmeans': {
        'n_clusters': 5,
        'random_state': RANDOM_STATE,
        'n_init': 10
    }
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# PCA components to retain
PCA_COMPONENTS = 20

# Feature selection threshold
FEATURE_CORRELATION_THRESHOLD = 0.95

# ============================================================================
# ALERT GENERATION
# ============================================================================

# Alert severity levels
ALERT_SEVERITY = {
    'Normal': 0,
    'Probe': 1,
    'DoS': 2,
    'R2L': 3,
    'U2R': 4
}

# Alert log format
ALERT_LOG_FILE = os.path.join(LOGS_DIR, 'intrusion_alerts.csv')
PREDICTION_LOG_FILE = os.path.join(LOGS_DIR, 'predictions.csv')

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (12, 8)
DPI = 100

# ============================================================================
# LOGGING
# ============================================================================

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'