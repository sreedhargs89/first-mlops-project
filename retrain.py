# retrain.py - Automated Model Retraining Script
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime
import os

def load_baseline_data():
    """Load original training data"""
    print("📥 Loading baseline training data...")
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    print(f"   ✅ Loaded {len(df)} baseline samples")
    print(f"   📊 Age range: {df['Age'].min()}-{df['Age'].max()}, Avg: {df['Age'].mean():.1f}")
    return df

def load_production_data(filepath="production_labeled.csv"):
    """Load production data with ground truth labels"""
    print("\n📥 Loading production data...")
    try:
        df = pd.read_csv(filepath)
        print(f"   ✅ Loaded {len(df)} production samples")
        print(f"   📊 Age range: {df['Age'].min()}-{df['Age'].max()}, Avg: {df['Age'].mean():.1f}")
        return df
    except FileNotFoundError:
        print(f"   ⚠️  {filepath} not found. Using simulated production data...")
        return generate_simulated_production_data()

def generate_simulated_production_data(n_samples=500):
    """Generate simulated production data for demo"""
    import numpy as np
    np.random.seed(42)
    
    # Simulate older population with different patterns
    data = {
        'Pregnancies': np.random.randint(0, 10, n_samples),
        'Glucose': np.random.normal(140, 30, n_samples),
        'BloodPressure': np.random.normal(72, 12, n_samples),
        'SkinThickness': np.random.normal(20, 15, n_samples),
        'Insulin': np.random.normal(80, 115, n_samples),
        'BMI': np.random.normal(34, 7, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.0, n_samples),
        'Age': np.random.randint(40, 80, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic outcomes based on risk factors
    # Higher age, glucose, BMI = higher diabetes probability
    risk_score = (
        (df['Age'] > 50).astype(int) * 0.3 +
        (df['Glucose'] > 140).astype(int) * 0.4 +
        (df['BMI'] > 35).astype(int) * 0.2 +
        np.random.uniform(0, 0.1, n_samples)
    )
    df['Outcome'] = (risk_score > 0.5).astype(int)
    
    return df

def combine_datasets(baseline_df, production_df):
    """Combine baseline and production data"""
    print("\n🔗 Combining datasets...")
    print(f"   Baseline: {len(baseline_df)} samples")
    print(f"   Production: {len(production_df)} samples")
    
    combined_df = pd.concat([baseline_df, production_df], ignore_index=True)
    
    print(f"   ✅ Combined: {len(combined_df)} total samples")
    print(f"   📊 New age distribution: {combined_df['Age'].min()}-{combined_df['Age'].max()}, Avg: {combined_df['Age'].mean():.1f}")
    
    return combined_df

def train_model(df, version="v2"):
    """Train new model on combined data"""
    print(f"\n🎓 Training new model ({version})...")
    
    # Prepare features
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
    X = df[feature_names]
    y = df["Outcome"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    
    print(f"\n📊 Model Performance:")
    print(f"   Training Accuracy: {train_accuracy:.2%}")
    print(f"   Testing Accuracy: {test_accuracy:.2%}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    
    # Feature importance
    print("\n🎯 Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    return model, test_accuracy

def save_model(model, version="v2"):
    """Save model with version and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as latest version
    model_path = f"diabetes_model_{version}.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # Also save with timestamp for history
    archive_path = f"models/diabetes_model_{version}_{timestamp}.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, archive_path)
    print(f"💾 Archived: {archive_path}")
    
    return model_path

def compare_models(old_model_path="diabetes_model.pkl", new_model_path="diabetes_model_v2.pkl"):
    """Compare old and new model performance"""
    print("\n📊 Comparing Models...")
    
    try:
        old_model = joblib.load(old_model_path)
        new_model = joblib.load(new_model_path)
        
        # Load test data (production data)
        test_data = generate_simulated_production_data(n_samples=200)
        X_test = test_data[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
        y_test = test_data["Outcome"]
        
        # Compare accuracy
        old_accuracy = old_model.score(X_test, y_test)
        new_accuracy = new_model.score(X_test, y_test)
        
        print(f"\n   Old Model (v1) Accuracy: {old_accuracy:.2%}")
        print(f"   New Model (v2) Accuracy: {new_accuracy:.2%}")
        print(f"   Improvement: {(new_accuracy - old_accuracy):.2%}")
        
        if new_accuracy > old_accuracy:
            print("\n   ✅ New model performs BETTER! Safe to deploy.")
        else:
            print("\n   ⚠️  New model performs WORSE! Investigate before deploying.")
            
    except FileNotFoundError as e:
        print(f"   ⚠️  Could not compare: {e}")

def main():
    """Main retraining pipeline"""
    print("="*60)
    print("🔄 AUTOMATED MODEL RETRAINING PIPELINE")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Load data
    baseline_data = load_baseline_data()
    production_data = load_production_data()
    
    # Step 2: Combine datasets
    combined_data = combine_datasets(baseline_data, production_data)
    
    # Step 3: Train new model
    new_model, accuracy = train_model(combined_data, version="v2")
    
    # Step 4: Save model
    model_path = save_model(new_model, version="v2")
    
    # Step 5: Compare with old model
    compare_models()
    
    print("\n" + "="*60)
    print("✅ RETRAINING COMPLETE!")
    print("="*60)
    print("\n📌 Next Steps:")
    print("   1. Review model performance metrics above")
    print("   2. Update Dockerfile to use new model")
    print("   3. Build new Docker image: docker build -t diabetes-api:v2 .")
    print("   4. Deploy to Kubernetes: kubectl set image deployment/diabetes-api ...")
    print("   5. Run monitor.py again to verify drift reduced")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
