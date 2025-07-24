import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class RandomForestTrainer:
    def __init__(self, csv_file='iot_dataS.csv'):
        self.csv_file = csv_file
        self.models_dir = 'models'
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
    def load_and_preprocess_data(self, sample_size=None):
        """Load and preprocess the IoT data WITHOUT removing anomaly data"""
        print("📊 Loading and preprocessing data...")
        
        try:
            df = pd.read_csv(self.csv_file)
            print(f"Original dataset shape: {df.shape}")
            
            # Sample data if specified
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
                print(f"Sampled dataset shape: {df.shape}")
            
            # Combine date and time into datetime
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            df = df.dropna(subset=['datetime'])
            df = df.sort_values('datetime')
            
            # Drop unnecessary columns
            df = df.drop(['date', 'time', 'epoch'], axis=1)
            
            # Encode status column
            df['status'] = df['status'].map({'normal': 0, 'anomaly': 1})
            
            # KEEP ALL DATA - both normal and anomaly
            print(f"Total data shape: {df.shape}")
            print(f"Normal samples: {sum(df['status'] == 0)}")
            print(f"Anomaly samples: {sum(df['status'] == 1)}")
            
            # Features for training
            features = ['moteid', 'temperature', 'humidity', 'light', 'voltage']
            X = df[features].values
            y = df['status'].values
            
            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            return X, X_scaled, y, scaler, df
            
        except Exception as e:
            print(f"❌ Data preprocessing error: {e}")
            return None, None, None, None, None
    
    def train_random_forest(self, X, y, is_test=False):
        """Train Random Forest model with both normal and anomaly data"""
        model_name = "rf_test.joblib" if is_test else "random_forest_model.joblib"
        
        print("🌲 Training Random Forest with anomaly data...")
        start_time = time.time()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set - Normal: {sum(y_train == 0)}, Anomaly: {sum(y_train == 1)}")
        print(f"Test set - Normal: {sum(y_test == 0)}, Anomaly: {sum(y_test == 1)}")
        
        # Train model with balanced parameters
        rf_model = RandomForestClassifier(
            n_estimators=200 if not is_test else 50,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        training_time = time.time() - start_time
        
        # Save model
        model_path = os.path.join(self.models_dir, model_name)
        joblib.dump(rf_model, model_path)
        
        print(f"✅ Random Forest trained in {training_time:.2f}s - Accuracy: {accuracy:.4f}")
        print(f"📦 Model saved as: {model_path}")
        
        # Print detailed classification report
        if not is_test:
            print("\n📊 Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
            
            print("\n📊 Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
            print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        return rf_model, accuracy, training_time
    
    def test_small_dataset(self):
        """Test training with small dataset first"""
        print("🧪 Testing with small dataset (10,000 rows)...")
        
        # Load small sample
        X, X_scaled, y, scaler, df = self.load_and_preprocess_data(sample_size=10000)
        
        if X is None:
            print("❌ Failed to load test data")
            return False
        
        try:
            # Test Random Forest
            rf_model, rf_acc, rf_time = self.train_random_forest(X, y, is_test=True)
            
            # Clean up test model
            test_file = 'models/rf_test.joblib'
            if os.path.exists(test_file):
                os.remove(test_file)
                print(f"🗑️ Removed test file: {test_file}")
            
            print("✅ Small dataset test passed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Small dataset test failed: {e}")
            return False
    
    def train_full_dataset(self):
        """Train Random Forest on full dataset"""
        print("🚀 Training Random Forest on full dataset...")
        
        # Load full dataset
        X, X_scaled, y, scaler, df = self.load_and_preprocess_data()
        
        if X is None:
            print("❌ Failed to load full data")
            return None
        
        results = {}
        
        try:
            # Train Random Forest
            rf_model, rf_acc, rf_time = self.train_random_forest(X, y)
            results['random_forest'] = {
                'model': rf_model,
                'accuracy': rf_acc,
                'training_time': rf_time
            }
            
            # Save scaler
            scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
            joblib.dump(scaler, scaler_path)
            print(f"📦 Scaler saved as: {scaler_path}")
            
            # Save dataset stats for simulator (including both normal and anomaly)
            stats = {
                'temperature': {
                    'mean': df['temperature'].mean(),
                    'std': df['temperature'].std(),
                    'min': df['temperature'].min(),
                    'max': df['temperature'].max()
                },
                'humidity': {
                    'mean': df['humidity'].mean(),
                    'std': df['humidity'].std(),
                    'min': df['humidity'].min(),
                    'max': df['humidity'].max()
                },
                'light': {
                    'mean': df['light'].mean(),
                    'std': df['light'].std(),
                    'min': df['light'].min(),
                    'max': df['light'].max()
                },
                'voltage': {
                    'mean': df['voltage'].mean(),
                    'std': df['voltage'].std(),
                    'min': df['voltage'].min(),
                    'max': df['voltage'].max()
                },
                'class_distribution': {
                    'normal': int(sum(df['status'] == 0)),
                    'anomaly': int(sum(df['status'] == 1))
                }
            }
            
            stats_path = os.path.join(self.models_dir, 'dataset_stats.joblib')
            joblib.dump(stats, stats_path)
            print(f"📊 Dataset stats saved as: {stats_path}")
            
            return results
            
        except Exception as e:
            print(f"❌ Full dataset training failed: {e}")
            return None
    
    def display_results(self, results):
        """Display training results"""
        if not results:
            return
        
        print("\n" + "="*60)
        print("📊 RANDOM FOREST TRAINING RESULTS (WITH ANOMALY DATA)")
        print("="*60)
        
        for model_name, result in results.items():
            if result['model'] is not None:
                print(f"\n🔹 {model_name.upper()}:")
                print(f"   Accuracy: {result['accuracy']:.4f}")
                print(f"   Training Time: {result['training_time']:.2f}s")
        
        print("\n✅ Random Forest model trained and saved successfully!")
        print("📁 Files saved:")
        print("   - models/random_forest_model.joblib")
        print("   - models/scaler.joblib") 
        print("   - models/dataset_stats.joblib")

def main():
    """Main training function"""
    print("🚀 Random Forest IoT Anomaly Detection Training (WITH ANOMALY DATA)")
    print("="*70)
    
    trainer = RandomForestTrainer()
    
    # Step 1: Test with small dataset
    if not trainer.test_small_dataset():
        print("❌ Small dataset test failed. Aborting full training.")
        return
    
    print("\n" + "="*70)
    
    # Step 2: Train on full dataset
    results = trainer.train_full_dataset()
    
    if results:
        trainer.display_results(results)
    else:
        print("❌ Full dataset training failed")

if __name__ == "__main__":
    main()