import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import joblib
import os

class EnhancedIoTSimulator:
    def __init__(self, num_records=5000):
        self.num_records = num_records
        self.load_dataset_stats()
        self.current_time = datetime(2004, 3, 1, 0, 0, 0)  # Start time
        self.moteid = 1
        
        # Enhanced trend control variables
        self.trend_duration = 0
        self.trend_direction = 'stable'
        self.trend_target = {}
        self.trend_intensity = 1.0  # How strong the trend is
        self.trend_randomness = 0.1  # Random variation in trend
        
        # Anomaly control variables
        self.anomaly_duration = 0
        self.anomaly_active = False
        self.anomaly_probability = 0.0001  # 0.01% chance as requested
        self.anomaly_type = 'none'  # 'temperature', 'voltage', 'combined'
        
        # Current sensor values (starting from realistic ranges)
        self.current_values = {
            'temperature': random.uniform(18.0, 28.0),  # Normal range start
            'humidity': random.uniform(30.0, 50.0),
            'light': random.uniform(50.0, 300.0),
            'voltage': random.uniform(2.4, 2.7)  # Normal voltage range
        }
        
        # Trend parameters - enhanced
        self.trend_change_rate = random.uniform(0.01, 0.05)  # Variable change rate
        self.max_trend_duration = 500  # Longer trends
        self.min_trend_duration = 100
        
        # Anomaly thresholds based on your data analysis
        self.anomaly_thresholds = {
            'temperature': {'min': -5.0, 'max': 45.0},  # Outside this range = anomaly
            'voltage': {'min': 2.27, 'max': 3.3},  # Below 3.3 can be anomaly
            'humidity': {'normal_min': 10.0, 'normal_max': 70.0},
            'light': {'normal_min': 0.0, 'normal_max': 1500.0}
        }
        
    def load_dataset_stats(self):
        """Load dataset statistics for realistic simulation"""
        try:
            stats_path = os.path.join('models', 'dataset_stats.joblib')
            if os.path.exists(stats_path):
                self.stats = joblib.load(stats_path)
                print("✅ Loaded dataset statistics")
            else:
                # Enhanced stats based on your actual dataset
                self.stats = {
                    'temperature': {'mean': 22.1, 'std': 3.3, 'min': -38.4, 'max': 385.6},
                    'humidity': {'mean': 39.3, 'std': 6.2, 'min': -8983.13, 'max': 137.5},
                    'light': {'mean': 415.4, 'std': 544.3, 'min': 0.0, 'max': 1847.4},
                    'voltage': {'mean': 2.56, 'std': 0.11, 'min': 0.009, 'max': 3.16}
                }
                print("⚠️ Using enhanced default statistics")
        except Exception as e:
            print(f"❌ Error loading stats: {e}")
            self.stats = {
                'temperature': {'mean': 22.1, 'std': 3.3, 'min': -38.4, 'max': 385.6},
                'humidity': {'mean': 39.3, 'std': 6.2, 'min': -8983.13, 'max': 137.5},
                'light': {'mean': 415.4, 'std': 544.3, 'min': 0.0, 'max': 1847.4},
                'voltage': {'mean': 2.56, 'std': 0.11, 'min': 0.009, 'max': 3.16}
            }
    
    def decide_trend_change(self):
        """Enhanced trend decision with directional persistence and random duration"""
        if self.trend_duration <= 0:
            # Start new trend with random duration
            self.trend_duration = random.randint(self.min_trend_duration, self.max_trend_duration)
            
            # Enhanced trend options with more variety
            trend_options = ['increase', 'decrease', 'stable', 'oscillate', 'drift_up', 'drift_down']
            self.trend_direction = random.choice(trend_options)
            
            # Set trend intensity (how strong the change will be)
            self.trend_intensity = random.uniform(0.5, 2.0)
            
            # Set trend targets with more sophisticated logic
            for param in ['temperature', 'humidity', 'light', 'voltage']:
                current = self.current_values[param]
                
                if self.trend_direction == 'increase':
                    change = random.uniform(2.0, 8.0) * self.trend_intensity
                    self.trend_target[param] = current + change
                elif self.trend_direction == 'decrease':
                    change = random.uniform(2.0, 8.0) * self.trend_intensity
                    self.trend_target[param] = current - change
                elif self.trend_direction == 'drift_up':
                    change = random.uniform(0.5, 2.0) * self.trend_intensity
                    self.trend_target[param] = current + change
                elif self.trend_direction == 'drift_down':
                    change = random.uniform(0.5, 2.0) * self.trend_intensity
                    self.trend_target[param] = current - change
                elif self.trend_direction == 'oscillate':
                    # Will oscillate around current value
                    self.trend_target[param] = current
                else:  # stable
                    self.trend_target[param] = current
                
                # Keep within reasonable bounds (not anomaly bounds)
                stats = self.stats[param]
                if param == 'temperature':
                    # Keep in normal range unless it's an anomaly
                    self.trend_target[param] = max(min(self.trend_target[param], 40.0), 0.0)
                elif param == 'voltage':
                    # Keep in normal voltage range
                    self.trend_target[param] = max(min(self.trend_target[param], 2.8), 2.3)
                else:
                    self.trend_target[param] = max(min(self.trend_target[param], 
                                                    stats['mean'] + 3 * stats['std']), 
                                                 stats['mean'] - 3 * stats['std'])
        
        self.trend_duration -= 1
    
    def decide_anomaly_state(self):
        """Enhanced anomaly decision with specific types"""
        if not self.anomaly_active:
            # Check if anomaly should start (very low probability as requested)
            if random.random() < self.anomaly_probability:
                self.anomaly_active = True
                self.anomaly_duration = random.randint(5, 30)  # Shorter anomaly periods
                
                # Decide type of anomaly based on your data analysis
                anomaly_types = ['temperature_high', 'temperature_low', 'voltage_drop', 
                               'combined_temp_voltage', 'sensor_failure']
                self.anomaly_type = random.choice(anomaly_types)
                
                print(f"🚨 Anomaly started - Type: {self.anomaly_type}, Duration: {self.anomaly_duration}")
        else:
            # Continue anomaly
            self.anomaly_duration -= 1
            if self.anomaly_duration <= 0:
                self.anomaly_active = False
                self.anomaly_type = 'none'
                print("✅ Anomaly ended")
    
    def generate_normal_value(self, param):
        """Enhanced normal value generation with better trends"""
        current = self.current_values[param]
        target = self.trend_target.get(param, current)
        
        # Calculate trend movement
        if self.trend_direction == 'oscillate':
            # Oscillating pattern
            oscillation = np.sin(self.trend_duration * 0.1) * random.uniform(0.5, 2.0)
            change = oscillation
        elif abs(current - target) > 0.1:
            # Move towards target with variable speed
            change_rate = self.trend_change_rate * random.uniform(0.5, 1.5)
            change = (target - current) * change_rate
        else:
            change = 0
        
        # Add realistic noise based on parameter type
        if param == 'temperature':
            noise = random.uniform(-0.2, 0.2)
        elif param == 'humidity':
            noise = random.uniform(-0.5, 0.5)
        elif param == 'light':
            noise = random.uniform(-2.0, 2.0)
        elif param == 'voltage':
            noise = random.uniform(-0.01, 0.01)
        else:
            noise = random.uniform(-0.1, 0.1)
        
        new_value = current + change + noise
        
        # Keep within normal operational bounds
        if param == 'temperature':
            new_value = max(min(new_value, 45.0), -5.0)  # Normal range
        elif param == 'voltage':
            new_value = max(min(new_value, 2.8), 2.3)  # Normal voltage range
        elif param == 'humidity':
            new_value = max(min(new_value, 80.0), 10.0)
        elif param == 'light':
            new_value = max(min(new_value, 1500.0), 0.0)
        
        return new_value
    
    def generate_anomaly_value(self, param):
        """Generate realistic anomalous values based on your data patterns"""
        current = self.current_values[param]
        
        if self.anomaly_type == 'temperature_high':
            if param == 'temperature':
                # High temperature anomaly (like 122.153 in your data)
                return random.uniform(80.0, 130.0)
            
        elif self.anomaly_type == 'temperature_low':
            if param == 'temperature':
                # Low temperature anomaly
                return random.uniform(-38.0, -10.0)
                
        elif self.anomaly_type == 'voltage_drop':
            if param == 'voltage':
                # Voltage drop anomaly (common in IoT sensors)
                return random.uniform(0.5, 2.0)
                
        elif self.anomaly_type == 'combined_temp_voltage':
            if param == 'temperature':
                return random.uniform(100.0, 125.0)  # High temp
            elif param == 'voltage':
                return random.uniform(1.8, 2.1)  # Low voltage
                
        elif self.anomaly_type == 'sensor_failure':
            if param == 'light':
                return 0  # Light sensor failure
            elif param == 'humidity':
                # Extreme humidity values like in your data
                if random.random() < 0.5:
                    return random.uniform(-50.0, -1.0)  # Negative humidity
                else:
                    return random.uniform(100.0, 130.0)  # Very high humidity
        
        # If not the parameter being affected by anomaly, return normal value
        return self.generate_normal_value(param)
    
    def is_anomalous(self, record):
        """Determine if a record should be labeled as anomaly based on thresholds"""
        temp = record['temperature']
        voltage = record['voltage']
        humidity = record['humidity']
        
        # Temperature outside normal range
        if temp < -5.0 or temp > 45.0:
            return True
            
        # Voltage anomalies (very low voltage)
        if voltage < 2.0:
            return True
            
        # Extreme humidity values
        if humidity < -10.0 or humidity > 100.0:
            return True
            
        # Combined conditions that indicate sensor malfunction
        if temp > 80.0 and voltage < 2.2:
            return True
            
        return False
    
    def generate_single_record(self):
        """Generate a single IoT sensor record with enhanced realism"""
        # Decide trend and anomaly state
        self.decide_trend_change()
        self.decide_anomaly_state()
        
        # Generate values based on current state
        for param in ['temperature', 'humidity', 'light', 'voltage']:
            if self.anomaly_active:
                self.current_values[param] = self.generate_anomaly_value(param)
            else:
                self.current_values[param] = self.generate_normal_value(param)
        
        # Create record
        record = {
            'date': self.current_time.strftime('%Y-%m-%d'),
            'time': self.current_time.strftime('%H:%M:%S.%f'),
            'epoch': int(self.current_time.timestamp()),
            'moteid': self.moteid,
            'temperature': round(self.current_values['temperature'], 6),
            'humidity': round(self.current_values['humidity'], 5),
            'light': round(self.current_values['light'], 2),
            'voltage': round(self.current_values['voltage'], 5),
            'status': ''  # Will be filled based on anomaly detection
        }
        
        # Keep status empty as requested - to be filled later
        record['status'] = ''
        
        # Increment time with variable intervals
        time_increment = random.randint(30, 180)  # 30 seconds to 3 minutes
        self.current_time += timedelta(seconds=time_increment)
        
        return record
    
    def generate_dataset(self):
        """Generate complete IoT dataset with enhanced features"""
        print(f"🔄 Generating {self.num_records} IoT sensor records...")
        print(f"📊 Anomaly probability: {self.anomaly_probability*100:.4f}%")
        
        records = []
        anomaly_generated = 0  # Track anomalies generated (not labeled)
        
        for i in range(self.num_records):
            record = self.generate_single_record()
            records.append(record)
            
            # Track when anomalies are generated (not labeled in status)
            if self.anomaly_active:
                anomaly_generated += 1
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"   Generated {i + 1}/{self.num_records} records... (Anomalies generated: {anomaly_generated})")
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Save to intermediate.csv
        df.to_csv('intermediate.csv', index=False)
        print(f"✅ Generated {len(df)} records saved to intermediate.csv")
        
        # Display enhanced statistics
        print("\n📊 Generated Dataset Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Anomalies generated: {anomaly_generated}")
        print(f"Anomaly generation rate: {(anomaly_generated/len(df)*100):.4f}%")
        print(f"Status field: Empty (to be filled by your processing pipeline)")
        print(f"Temperature range: {df['temperature'].min():.2f} to {df['temperature'].max():.2f}")
        print(f"Humidity range: {df['humidity'].min():.2f} to {df['humidity'].max():.2f}")
        print(f"Light range: {df['light'].min():.2f} to {df['light'].max():.2f}")
        print(f"Voltage range: {df['voltage'].min():.5f} to {df['voltage'].max():.5f}")
        print(f"Time span: {df['date'].min()} to {df['date'].max()}")
        
        # Show sample records with potential anomalies (but status empty)
        if anomaly_generated > 0:
            print(f"\n📋 Sample records (status empty for your processing):")
            print(df[['temperature', 'humidity', 'light', 'voltage', 'status']].head(10))
        
        return df

def main():
    """Main simulation function"""
    print("🚀 Enhanced IoT Sensor Data Simulator")
    print("="*50)
    print("Features:")
    print("- Realistic trend patterns with random durations")
    print("- Enhanced anomaly detection (0.01% probability)")
    print("- Temperature anomalies outside -5°C to 45°C")
    print("- Voltage anomalies below normal range")
    print("- Combined sensor failure scenarios")
    print("="*50)
    
    # Get number of records to generate
    try:
        num_records = int(input("Enter number of records to generate (default 5000): ") or 5000)
    except ValueError:
        num_records = 5000
        print("Using default: 5000 records")
    
    # Create simulator and generate data
    simulator = EnhancedIoTSimulator(num_records)
    dataset = simulator.generate_dataset()
    
    print("\n✅ Enhanced simulation completed successfully!")
    print("📁 Data saved to: intermediate.csv")
    print("🔄 Next step: Run preprocessing and model inference")
    print("\n💡 Key enhancements:")
    print("   - Directional trends with random durations")
    print("   - Realistic anomaly patterns matching your dataset")
    print("   - Temperature and voltage-based anomaly detection")
    print("   - Variable trend intensity and oscillating patterns")

if __name__ == "__main__":
    main()