"""
SentinelNet - Milestone Execution Script
Interactive script to run specific project milestones

Milestones:
- Milestone 1: Data Acquisition and Preprocessing (Weeks 1-2)
- Milestone 2: Feature Engineering and Supervised Training (Weeks 3-4)
- Milestone 3: Anomaly Detection and Tuning (Weeks 5-6)
- Milestone 4: Deployment and Documentation (Weeks 7-8)
"""

import sys
import logging
from datetime import datetime
from sentinelnet_main import SentinelNetNIDS
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner"""
    print("\n" + "=" * 80)
    print("  SENTINELNET - MILESTONE EXECUTION")
    print("  AI-Powered Network Intrusion Detection System")
    print("=" * 80 + "\n")


def print_milestone_menu():
    """Display milestone selection menu"""
    print("\nAvailable Milestones:")
    print("-" * 80)
    print("  [1] Milestone 1: Data Acquisition and Preprocessing (Weeks 1-2)")
    print("      - Download NSL-KDD dataset")
    print("      - Explore and analyze data")
    print("      - Clean and preprocess features")
    print()
    print("  [2] Milestone 2: Feature Engineering and Supervised Training (Weeks 3-4)")
    print("      - Perform feature engineering")
    print("      - Train supervised models (RF, SVM, Logistic Regression)")
    print("      - Initial model evaluation")
    print()
    print("  [3] Milestone 3: Anomaly Detection and Tuning (Weeks 5-6)")
    print("      - Implement unsupervised learning")
    print("      - Model evaluation and comparison")
    print("      - Hyperparameter tuning")
    print()
    print("  [4] Milestone 4: Deployment and Documentation (Weeks 7-8)")
    print("      - Alert generation and logging")
    print("      - Generate comprehensive report")
    print("      - Save all models")
    print()
    print("  [A] Run ALL Milestones (Complete Pipeline)")
    print("  [Q] Quit")
    print("-" * 80)


def run_milestone_1(nids):
    """Execute Milestone 1: Data Acquisition and Preprocessing"""
    logger.info("\n" + "=" * 80)
    logger.info("MILESTONE 1: Data Acquisition and Preprocessing")
    logger.info("Weeks 1-2")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Week 1: Dataset Acquisition and Exploration
        logger.info("\nWeek 1: Dataset Acquisition and Exploration")
        nids.download_dataset()
        nids.load_and_explore_data()
        
        # Week 2: Data Cleaning and Preprocessing
        logger.info("\nWeek 2: Data Cleaning and Preprocessing")
        nids.preprocess_data()
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✓ Milestone 1 completed in {duration:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Milestone 1 failed: {e}")
        return False


def run_milestone_2(nids):
    """Execute Milestone 2: Feature Engineering and Supervised Training"""
    logger.info("\n" + "=" * 80)
    logger.info("MILESTONE 2: Feature Engineering and Supervised Training")
    logger.info("Weeks 3-4")
    logger.info("=" * 80)
    
    # Ensure Milestone 1 data is available
    if nids.X_train is None:
        logger.info("Milestone 1 data not found. Running prerequisite steps...")
        if not run_milestone_1(nids):
            return False
    
    start_time = datetime.now()
    
    try:
        # Week 3: Feature Engineering
        logger.info("\nWeek 3: Feature Engineering and Selection")
        nids.feature_engineering()
        
        # Week 4: Supervised Model Training
        logger.info("\nWeek 4: Supervised Model Training")
        nids.train_supervised_models()
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✓ Milestone 2 completed in {duration:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Milestone 2 failed: {e}")
        return False


def run_milestone_3(nids):
    """Execute Milestone 3: Anomaly Detection and Tuning"""
    logger.info("\n" + "=" * 80)
    logger.info("MILESTONE 3: Anomaly Detection and Tuning")
    logger.info("Weeks 5-6")
    logger.info("=" * 80)
    
    # Ensure previous milestones are complete
    if nids.X_train is None or len(nids.models) == 0:
        logger.info("Previous milestone data not found. Running prerequisite steps...")
        if not run_milestone_2(nids):
            return False
    
    start_time = datetime.now()
    
    try:
        # Week 5: Anomaly Detection
        logger.info("\nWeek 5: Anomaly Detection with Unsupervised Learning")
        nids.anomaly_detection()
        
        # Week 6: Model Evaluation and Tuning
        logger.info("\nWeek 6: Model Evaluation and Fine-tuning")
        nids.evaluate_models()
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✓ Milestone 3 completed in {duration:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Milestone 3 failed: {e}")
        return False


def run_milestone_4(nids):
    """Execute Milestone 4: Deployment and Documentation"""
    logger.info("\n" + "=" * 80)
    logger.info("MILESTONE 4: Deployment and Documentation")
    logger.info("Weeks 7-8")
    logger.info("=" * 80)
    
    # Ensure all previous milestones are complete
    if nids.best_model is None:
        logger.info("Previous milestone data not found. Running prerequisite steps...")
        if not run_milestone_3(nids):
            return False
    
    start_time = datetime.now()
    
    try:
        # Week 7: Alert Generation and Logging
        logger.info("\nWeek 7: Alert Generation and Logging")
        nids.generate_alerts(sample_size=200)
        
        # Week 8: Documentation and Presentation
        logger.info("\nWeek 8: Documentation and Presentation Preparation")
        nids.generate_report()
        nids.save_models()
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✓ Milestone 4 completed in {duration:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Milestone 4 failed: {e}")
        return False


def run_all_milestones(nids):
    """Execute all milestones sequentially"""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING ALL MILESTONES")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    milestones = [
        ("Milestone 1", run_milestone_1),
        ("Milestone 2", run_milestone_2),
        ("Milestone 3", run_milestone_3),
        ("Milestone 4", run_milestone_4)
    ]
    
    for name, milestone_func in milestones:
        if not milestone_func(nids):
            logger.error(f"\n✗ Pipeline stopped at {name}")
            return False
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 80)
    logger.info(f"ALL MILESTONES COMPLETED in {duration:.2f} seconds")
    logger.info("=" * 80)
    
    return True


def main():
    """Main interactive menu"""
    print_banner()
    
    # Initialize NIDS
    nids = SentinelNetNIDS()
    
    while True:
        print_milestone_menu()
        
        choice = input("\nSelect milestone to run [1-4/A/Q]: ").strip().upper()
        
        if choice == 'Q':
            print("\nExiting... Goodbye!")
            break
        elif choice == '1':
            run_milestone_1(nids)
        elif choice == '2':
            run_milestone_2(nids)
        elif choice == '3':
            run_milestone_3(nids)
        elif choice == '4':
            run_milestone_4(nids)
        elif choice == 'A':
            run_all_milestones(nids)
            break  # Exit after running all
        else:
            print("\n✗ Invalid choice. Please select 1-4, A, or Q.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()