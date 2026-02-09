"""
SentinelNet - Quick Project Execution
Single command to run the complete NIDS pipeline
"""

import sys
import logging
from datetime import datetime
from sentinelnet_main import SentinelNetNIDS

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def main():
    """Execute the complete SentinelNet NIDS pipeline"""
    print("\n" + "=" * 80)
    print("  SENTINELNET - AI-POWERED NETWORK INTRUSION DETECTION SYSTEM")
    print("  Complete Pipeline Execution")
    print("=" * 80 + "\n")
    
    print("This will execute all 8 modules:")
    print("  ✓ Module 1: Dataset Acquisition and Exploration")
    print("  ✓ Module 2: Data Cleaning and Preprocessing")
    print("  ✓ Module 3: Feature Engineering and Selection")
    print("  ✓ Module 4: Model Building and Training (Supervised)")
    print("  ✓ Module 5: Anomaly Detection (Unsupervised)")
    print("  ✓ Module 6: Model Evaluation and Tuning")
    print("  ✓ Module 7: Alert Generation and Logging")
    print("  ✓ Module 8: Documentation and Presentation\n")
    
    start_time = datetime.now()
    
    try:
        # Initialize and run
        nids = SentinelNetNIDS()
        nids.run_full_pipeline()
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        print("\n" + "=" * 80)
        print("  ✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"  Total execution time: {minutes}m {seconds}s")
        print("=" * 80)
        print("\n📁 Generated Outputs:")
        print(f"  • Models: models/")
        print(f"  • Results: results/")
        print(f"  • Logs: logs/")
        print(f"  • Alerts: logs/intrusion_alerts.csv")
        print(f"  • Report: results/project_report.txt")
        print("\n" + "=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())