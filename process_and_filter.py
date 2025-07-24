import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    print("✅ TensorFlow imported successfully")
except ImportError as e:
    print(f"❌ TensorFlow import error: {e}")
    exit(1)

class EnhancedIoTDataProcessor:
    def __init__(self):
        self.models_loaded = False
        self.rf_model = None
        self.scaler = None
        self.trend_scaler = MinMaxScaler(feature_range=(0, 1))
        self.trend_window_size = 20
        self.load_models()
        
    def load_models(self):
        """Load trained Random Forest model and scaler"""
        print("📦 Loading trained models...")
        try:
            # Load Random Forest model and scaler
            self.rf_model = joblib.load('models/random_forest_model.joblib')
            self.scaler = joblib.load('models/scaler.joblib')
            self.models_loaded = True
            print("✅ Random Forest model and scaler loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False
    
    def build_trend_lstm(self):
        """Build LSTM model for trend extraction"""
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(self.trend_window_size, 4)),
            LSTM(16),
            Dense(4)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def preprocess_data(self, csv_file='intermediate.csv'):
        """Load and preprocess the intermediate data"""
        print("🔄 Loading intermediate.csv...")
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 Loaded {len(df)} records from {csv_file}")
            
            # Create datetime column for sorting
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
            
            # Prepare features for model prediction
            features = ['moteid', 'temperature', 'humidity', 'light', 'voltage']
            X = df[features].values
            X_scaled = self.scaler.transform(X)
            
            # Prepare features for trend analysis
            trend_features = ['temperature', 'humidity', 'light', 'voltage']
            trend_data = df[trend_features].values
            
            return df, X, X_scaled, trend_data
        except Exception as e:
            print(f"❌ Error preprocessing data: {e}")
            return None, None, None, None
    
    def create_trend_sequences(self, data, start_idx, end_idx):
        """Create sequences for trend LSTM from a specific range"""
        if end_idx - start_idx < self.trend_window_size:
            return None, None
        
        segment_data = data[start_idx:end_idx]
        scaled_segment = self.trend_scaler.fit_transform(segment_data)
        
        X, y = [], []
        for i in range(len(scaled_segment) - self.trend_window_size):
            X.append(scaled_segment[i:i+self.trend_window_size])
            y.append(scaled_segment[i+self.trend_window_size])
        
        return np.array(X), np.array(y)
    
    def extract_trend_points(self, data, start_idx, end_idx):
        """Extract trend points from a data segment using LSTM"""
        try:
            X, y = self.create_trend_sequences(data, start_idx, end_idx)
            
            if X is None or len(X) < 10:  # Need minimum data for trend analysis
                return []
            
            # Build and train trend LSTM
            trend_model = self.build_trend_lstm()
            trend_model.fit(X, y, epochs=5, batch_size=min(32, len(X)), verbose=0)
            
            # Predict and calculate reconstruction errors
            predicted = trend_model.predict(X, verbose=0)
            predicted = self.trend_scaler.inverse_transform(predicted)
            actual = self.trend_scaler.inverse_transform(y)
            
            errors = np.mean(np.abs(predicted - actual), axis=1)
            
            # Find trend points using error peaks
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(errors, distance=10, prominence=np.std(errors))
            except ImportError:
                # Fallback if scipy not available - use simple peak detection
                peaks = []
                for i in range(1, len(errors)-1):
                    if errors[i] > errors[i-1] and errors[i] > errors[i+1] and errors[i] > np.mean(errors):
                        peaks.append(i)
            
            # Convert to original dataframe indices
            trend_indices = [start_idx + self.trend_window_size + i for i in peaks]
            
            # Always include start and end of segment
            if start_idx not in trend_indices:
                trend_indices.insert(0, start_idx)
            if (end_idx - 1) not in trend_indices:
                trend_indices.append(end_idx - 1)
            
            return sorted(list(set(trend_indices)))
            
        except Exception as e:
            print(f"⚠️ Trend extraction failed for segment {start_idx}-{end_idx}: {e}")
            return [start_idx, end_idx - 1] if end_idx > start_idx else []
    
    def process_with_enhanced_pipeline(self, df, X, X_scaled, trend_data):
        """Enhanced processing pipeline with trend detection"""
        print("🚀 Starting enhanced data processing pipeline...")
        
        # Step 1: Run Random Forest on all data
        print("🌲 Running Random Forest predictions...")
        rf_predictions = self.rf_model.predict(X)
        
        # Add RF predictions to dataframe
        df['rf_prediction'] = rf_predictions
        df['rf_status'] = df['rf_prediction'].map({0: 'normal', 1: 'anomaly'})
        
        blockchain_data = []
        in_anomaly_sequence = False
        normal_segment_start = None
        
        # Initialize tracking columns
        for col in ['stored_in_blockchain', 'skip_reason', 'storage_type']:
            df[col] = [False] * len(df) if col == 'stored_in_blockchain' else [''] * len(df)
        
        print("⚡ Processing data through enhanced pipeline logic...")
        
        i = 0
        while i < len(df):
            current_row = df.iloc[i]
            rf_pred = rf_predictions[i]
            
            if rf_pred == 1:  # Anomaly detected by RF
                # Process any pending normal segment before handling anomaly
                if normal_segment_start is not None and i > normal_segment_start:
                    trend_indices = self.extract_trend_points(trend_data, normal_segment_start, i)
                    self.store_trend_points(df, trend_indices, blockchain_data, 'trend_normal')
                    normal_segment_start = None
                
                if not in_anomaly_sequence:
                    # First anomaly in sequence - store directly to blockchain
                    self.store_single_record(df, i, blockchain_data, 'anomaly', 'RF_first_anomaly')
                    print(f"🚨 First anomaly detected at {current_row['datetime']} - stored to blockchain")
                    in_anomaly_sequence = True
                else:
                    # Consecutive anomaly - skip storing
                    df.loc[i, 'skip_reason'] = 'consecutive_anomaly'
                    print(f"⏭️ Consecutive anomaly at {current_row['datetime']} - skipped")
                
            else:  # Normal data according to RF
                if in_anomaly_sequence:
                    # End of anomaly sequence, start new normal segment
                    in_anomaly_sequence = False
                    normal_segment_start = i
                    print(f"✅ Normal data resumed at {current_row['datetime']} - starting trend analysis")
                elif normal_segment_start is None:
                    # First normal data or start of processing
                    normal_segment_start = i
            
            i += 1
        
        # Process final normal segment if exists
        if normal_segment_start is not None and normal_segment_start < len(df):
            trend_indices = self.extract_trend_points(trend_data, normal_segment_start, len(df))
            self.store_trend_points(df, trend_indices, blockchain_data, 'trend_normal')
        
        print(f"📈 Enhanced pipeline completed:")
        print(f"   - Total records processed: {len(df)}")
        print(f"   - Total records stored in blockchain: {len(blockchain_data)}")
        
        return df, blockchain_data
    
    def store_single_record(self, df, index, blockchain_data, status, source):
        """Store a single record to blockchain"""
        row = df.iloc[index]
        blockchain_record = {
            'date': row['date'],
            'time': row['time'],
            'epoch': row['epoch'],
            'moteid': row['moteid'],
            'temperature': row['temperature'],
            'humidity': row['humidity'],
            'light': row['light'],
            'voltage': row['voltage'],
            'status': status,
            'source': source
        }
        blockchain_data.append(blockchain_record)
        df.loc[index, 'stored_in_blockchain'] = True
        df.loc[index, 'storage_type'] = source
    
    def store_trend_points(self, df, trend_indices, blockchain_data, storage_type):
        """Store trend points to blockchain"""
        if not trend_indices:
            return
        
        print(f"📊 Storing {len(trend_indices)} trend points to blockchain")
        
        for idx in trend_indices:
            if idx < len(df):
                self.store_single_record(df, idx, blockchain_data, 'normal', storage_type)
    
    def save_results(self, processed_df, blockchain_data):
        """Save results to files"""
        print("💾 Saving results...")
        
        try:
            # Save processed intermediate data with predictions
            processed_df.to_csv('intermediate_processed.csv', index=False)
            print(f"📄 Saved intermediate_processed.csv ({len(processed_df)} records)")
            
            # Save blockchain data
            if blockchain_data:
                blockchain_df = pd.DataFrame(blockchain_data)
                blockchain_df.to_csv('blockchain.csv', index=False)
                print(f"⛓️ Saved blockchain.csv ({len(blockchain_df)} records)")
            else:
                print("⚠️ No data qualified for blockchain storage")
                # Create empty blockchain file
                pd.DataFrame(columns=['date', 'time', 'epoch', 'moteid', 'temperature', 
                                    'humidity', 'light', 'voltage', 'status', 'source']).to_csv('blockchain.csv', index=False)
                
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def generate_enhanced_report(self, processed_df, blockchain_data):
        """Generate detailed comparison report with trend analysis"""
        print("📊 Generating enhanced comparison report...")
        
        # Original data analysis
        original_total = len(processed_df)
        
        # RF analysis
        rf_anomalies = processed_df['rf_prediction'].sum()
        rf_normal = original_total - rf_anomalies
        
        # Blockchain analysis
        blockchain_total = len(blockchain_data)
        blockchain_anomalies = sum(1 for record in blockchain_data if record['status'] == 'anomaly')
        blockchain_normal = blockchain_total - blockchain_anomalies
        
        # Storage type analysis
        trend_stored = processed_df['storage_type'].str.contains('trend', na=False).sum()
        anomaly_stored = processed_df['storage_type'].str.contains('anomaly', na=False).sum()
        
        # Processing analysis
        consecutive_skipped = (processed_df['skip_reason'] == 'consecutive_anomaly').sum()
        
        # Data reduction analysis
        data_reduction = ((original_total - blockchain_total) / original_total * 100) if original_total > 0 else 0
        
        report = f"""
Enhanced IoT Data Processing & Trend Analysis Report
==================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ORIGINAL DATA (intermediate.csv):
- Total records: {original_total:,}

RANDOM FOREST ANALYSIS:
- Detected anomalies: {rf_anomalies:,} ({rf_anomalies/original_total*100:.1f}%)
- Detected normal: {rf_normal:,} ({rf_normal/original_total*100:.1f}%)

BLOCKCHAIN STORAGE (blockchain.csv):
- Total stored records: {blockchain_total:,}
- Anomaly records stored: {blockchain_anomalies:,}
- Normal trend points stored: {blockchain_normal:,}

STORAGE BREAKDOWN:
- First anomalies stored: {anomaly_stored:,}
- Trend points stored: {trend_stored:,}
- Consecutive anomalies skipped: {consecutive_skipped:,}

DATA REDUCTION EFFICIENCY:
- Original to Blockchain reduction: {data_reduction:.1f}%
- Storage efficiency: {blockchain_total/original_total*100:.1f}% of original data stored

ENHANCED PIPELINE FEATURES:
✅ Dynamic trend detection using LSTM on normal data segments
✅ Automatic window reset after anomaly sequences
✅ Real-time trend point identification
✅ Optimal blockchain storage with trend preservation
✅ Significant data reduction while maintaining critical information

TREND ANALYSIS BENEFITS:
- Captures key data patterns without storing redundant information
- Maintains temporal relationships in stored data
- Reduces blockchain overhead by {data_reduction:.1f}%
- Preserves both anomalies and normal data trends

PIPELINE EFFECTIVENESS:
The enhanced processing pipeline successfully:
1. Identified {rf_anomalies:,} anomalous data points using Random Forest
2. Stored only the first anomaly in sequences (avoiding redundancy)
3. Applied dynamic LSTM trend analysis on normal data segments
4. Extracted {trend_stored:,} critical trend points from normal data
5. Achieved {data_reduction:.1f}% data reduction while preserving both anomalies and trends
6. Optimized blockchain storage with {blockchain_total:,} essential records

This demonstrates effective real-time anomaly detection and trend analysis 
for optimized IoT blockchain storage.
"""
        
        # Save report
        with open('enhanced_processing_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("📋 Generated enhanced_processing_report.txt")
        
        # Print summary to console
        print("\n" + "="*70)
        print("📊 ENHANCED PROCESSING SUMMARY")
        print("="*70)
        print(f"Original data: {original_total:,} records")
        print(f"Blockchain data: {blockchain_total:,} records")
        print(f"Data reduction: {data_reduction:.1f}%")
        print(f"Anomalies stored: {blockchain_anomalies:,}")
        print(f"Trend points stored: {blockchain_normal:,}")
        print(f"RF anomalies detected: {rf_anomalies:,}")
        print("="*70)
    
    def run_enhanced_pipeline(self):
        """Execute the complete enhanced processing pipeline"""
        if not self.models_loaded:
            print("❌ Cannot proceed - models not loaded properly")
            return False
        
        # Check for input file
        if not os.path.exists('intermediate.csv'):
            print("❌ Error: intermediate.csv not found")
            return False
        
        print("\n" + "="*70)
        print("🚀 ENHANCED IoT DATA PROCESSING & TREND ANALYSIS PIPELINE")
        print("="*70)
        
        # Step 1: Load and preprocess data
        df, X, X_scaled, trend_data = self.preprocess_data()
        if df is None:
            print("❌ Failed to preprocess data")
            return False
        
        # Step 2: Process through enhanced RF and LSTM pipeline
        processed_df, blockchain_data = self.process_with_enhanced_pipeline(df, X, X_scaled, trend_data)
        
        # Step 3: Save results
        self.save_results(processed_df, blockchain_data)
        
        # Step 4: Generate enhanced report
        self.generate_enhanced_report(processed_df, blockchain_data)
        
        print("\n✅ Enhanced pipeline execution completed successfully!")
        print("\nGenerated files:")
        print("- intermediate_processed.csv (detailed processing results)")
        print("- blockchain.csv (filtered data with anomalies and trends)")
        print("- enhanced_processing_report.txt (comprehensive analysis)")
        
        return True

def main():
    """Main function to run the enhanced processing pipeline"""
    
    # Check for required model files (only RF model needed now)
    required_files = [
        'models/random_forest_model.joblib',
        'models/scaler.joblib',
        'intermediate.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure Random Forest model, scaler, and intermediate.csv are available.")
        print("Note: Pre-trained LSTM model is no longer required - trend LSTM is built dynamically.")
        return
    
    # Initialize and run enhanced processor
    processor = EnhancedIoTDataProcessor()
    success = processor.run_enhanced_pipeline()
    
    if not success:
        print("❌ Enhanced pipeline execution failed")
    else:
        print("\n🎉 All enhanced processing completed successfully!")
        print("\n🔥 Key Improvements:")
        print("   ✅ Dynamic LSTM trend detection on normal data segments")
        print("   ✅ Automatic training window reset after anomalies")
        print("   ✅ Real-time trend point extraction")
        print("   ✅ Optimal blockchain storage with maximum data reduction")

if __name__ == "__main__":
    main()