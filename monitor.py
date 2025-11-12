# monitor.py - Data Drift Detection for Diabetes Prediction Model
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataDriftDetector:
    """
    Monitors data drift using statistical tests (KS test, PSI)
    Compares production data against training baseline
    """
    
    def __init__(self, baseline_data_path: str = None, model_path: str = "diabetes_model.pkl"):
        self.model = joblib.load(model_path)
        self.feature_names = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
        
        # Load or create baseline statistics
        if baseline_data_path:
            self.baseline_df = pd.read_csv(baseline_data_path)
            self.baseline_stats = self._calculate_baseline_stats(self.baseline_df)
        else:
            # Use default baseline from training data
            self.baseline_stats = self._load_default_baseline()
    
    def _load_default_baseline(self) -> Dict:
        """Load baseline statistics from training dataset"""
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        df = pd.read_csv(url)
        baseline_df = df[self.feature_names]
        return self._calculate_baseline_stats(baseline_df)
    
    def _calculate_baseline_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical measures for baseline data"""
        stats_dict = {}
        for feature in self.feature_names:
            stats_dict[feature] = {
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max(),
                'q25': df[feature].quantile(0.25),
                'q50': df[feature].quantile(0.50),
                'q75': df[feature].quantile(0.75),
                'distribution': df[feature].values
            }
        return stats_dict
    
    def kolmogorov_smirnov_test(self, production_data: pd.DataFrame, 
                                alpha: float = 0.05) -> Dict:
        """
        Perform KS test to detect distribution changes
        Returns drift status for each feature
        """
        drift_results = {}
        
        for feature in self.feature_names:
            baseline_dist = self.baseline_stats[feature]['distribution']
            production_dist = production_data[feature].values
            
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(baseline_dist, production_dist)
            
            drift_detected = p_value < alpha
            
            drift_results[feature] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift_detected': bool(drift_detected),
                'baseline_mean': float(self.baseline_stats[feature]['mean']),
                'production_mean': float(production_data[feature].mean()),
                'mean_shift': float(production_data[feature].mean() - 
                                  self.baseline_stats[feature]['mean'])
            }
        
        return drift_results
    
    def population_stability_index(self, production_data: pd.DataFrame, 
                                  bins: int = 10) -> Dict:
        """
        Calculate PSI (Population Stability Index)
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change (retrain recommended)
        """
        psi_results = {}
        
        for feature in self.feature_names:
            baseline_dist = self.baseline_stats[feature]['distribution']
            production_dist = production_data[feature].values
            
            # Create bins based on baseline
            min_val = min(baseline_dist.min(), production_dist.min())
            max_val = max(baseline_dist.max(), production_dist.max())
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate distributions
            baseline_counts, _ = np.histogram(baseline_dist, bins=bin_edges)
            production_counts, _ = np.histogram(production_dist, bins=bin_edges)
            
            # Avoid division by zero
            baseline_pct = (baseline_counts + 1) / (len(baseline_dist) + bins)
            production_pct = (production_counts + 1) / (len(production_dist) + bins)
            
            # Calculate PSI
            psi = np.sum((production_pct - baseline_pct) * 
                        np.log(production_pct / baseline_pct))
            
            # Determine severity
            if psi < 0.1:
                severity = "No significant change"
                action = "Continue monitoring"
            elif psi < 0.2:
                severity = "Moderate change"
                action = "Investigate and prepare for retraining"
            else:
                severity = "Significant change"
                action = "Retrain model immediately"
            
            psi_results[feature] = {
                'psi': float(psi),
                'severity': severity,
                'action': action
            }
        
        return psi_results
    
    def detect_drift(self, production_data: pd.DataFrame) -> Dict:
        """
        Main drift detection method combining multiple tests
        """
        print(f"\n{'='*60}")
        print(f"🔍 DATA DRIFT DETECTION REPORT")
        print(f"{'='*60}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Production samples: {len(production_data)}")
        print(f"{'='*60}\n")
        
        # Run KS test
        ks_results = self.kolmogorov_smirnov_test(production_data)
        
        # Run PSI
        psi_results = self.population_stability_index(production_data)
        
        # Generate summary
        drift_detected_features = [f for f, v in ks_results.items() 
                                  if v['drift_detected']]
        high_psi_features = [f for f, v in psi_results.items() 
                           if v['psi'] >= 0.2]
        
        # Print detailed results
        print("📊 KOLMOGOROV-SMIRNOV TEST RESULTS:")
        print("-" * 60)
        for feature, result in ks_results.items():
            status = "⚠️  DRIFT DETECTED" if result['drift_detected'] else "✅ No drift"
            print(f"\n{feature}:")
            print(f"  Status: {status}")
            print(f"  KS Statistic: {result['ks_statistic']:.4f}")
            print(f"  P-value: {result['p_value']:.4f}")
            print(f"  Baseline Mean: {result['baseline_mean']:.2f}")
            print(f"  Production Mean: {result['production_mean']:.2f}")
            print(f"  Mean Shift: {result['mean_shift']:+.2f}")
        
        print(f"\n{'='*60}")
        print("📈 POPULATION STABILITY INDEX (PSI) RESULTS:")
        print("-" * 60)
        for feature, result in psi_results.items():
            print(f"\n{feature}:")
            print(f"  PSI: {result['psi']:.4f}")
            print(f"  Severity: {result['severity']}")
            print(f"  Action: {result['action']}")
        
        print(f"\n{'='*60}")
        print("🎯 SUMMARY & RECOMMENDATIONS:")
        print("-" * 60)
        
        overall_drift = len(drift_detected_features) > 0 or len(high_psi_features) > 0
        
        if overall_drift:
            print("⚠️  DATA DRIFT DETECTED!")
            if drift_detected_features:
                print(f"   KS Test flagged: {', '.join(drift_detected_features)}")
            if high_psi_features:
                print(f"   High PSI flagged: {', '.join(high_psi_features)}")
            print("\n📌 RECOMMENDED ACTIONS:")
            print("   1. Investigate data quality issues")
            print("   2. Review recent data collection changes")
            print("   3. Prepare for model retraining")
            print("   4. Consider A/B testing with retrained model")
        else:
            print("✅ No significant drift detected")
            print("   Model performance expected to be stable")
            print("   Continue regular monitoring")
        
        print(f"{'='*60}\n")
        
        # Return structured results
        return {
            'timestamp': datetime.now().isoformat(),
            'production_samples': len(production_data),
            'overall_drift_detected': overall_drift,
            'ks_test_results': ks_results,
            'psi_results': psi_results,
            'drift_features_ks': drift_detected_features,
            'high_psi_features': high_psi_features,
            'recommendation': "RETRAIN" if overall_drift else "MONITOR"
        }
    
    def save_report(self, results: Dict, filepath: str = "drift_report.json"):
        """Save drift detection results to file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Report saved to {filepath}")


def load_production_data(filepath: str) -> pd.DataFrame:
    """Load production data from CSV or generate sample"""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"⚠️  File {filepath} not found. Using simulated data for demo.")
        return generate_sample_production_data()


def generate_sample_production_data(n_samples: int = 100, 
                                    add_drift: bool = True) -> pd.DataFrame:
    """Generate sample production data with optional drift"""
    np.random.seed(42)
    
    if add_drift:
        # Simulate drift: higher glucose, BMI, age
        data = {
            'Pregnancies': np.random.randint(0, 10, n_samples),
            'Glucose': np.random.normal(140, 30, n_samples),  # Higher mean (was ~120)
            'BloodPressure': np.random.normal(72, 12, n_samples),
            'BMI': np.random.normal(34, 7, n_samples),  # Higher mean (was ~32)
            'Age': np.random.randint(30, 80, n_samples)  # Older population
        }
    else:
        # No drift: similar to training data
        data = {
            'Pregnancies': np.random.randint(0, 10, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples),
            'BloodPressure': np.random.normal(70, 12, n_samples),
            'BMI': np.random.normal(32, 7, n_samples),
            'Age': np.random.randint(21, 70, n_samples)
        }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("🚀 Starting Data Drift Detection...")
    
    # Initialize detector
    detector = DataDriftDetector()
    
    # Load or generate production data
    # Option 1: Load from file
    # production_data = load_production_data("production_predictions.csv")
    
    # Option 2: Generate sample data with drift for demo
    print("\n📊 Generating sample production data with simulated drift...")
    production_data = generate_sample_production_data(n_samples=200, add_drift=True)
    
    # Detect drift
    results = detector.detect_drift(production_data)
    
    # Save report
    detector.save_report(results)
    
    # Example: Check if retraining is needed
    if results['recommendation'] == "RETRAIN":
        print("\n🔄 Triggering automated retraining pipeline...")
        print("   (Pipeline integration to be implemented)")
    else:
        print("\n✅ No action needed. Model is performing well.")
