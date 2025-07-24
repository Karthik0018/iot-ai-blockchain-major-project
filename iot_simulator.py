import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import joblib
import os

class IoTSimulator:
    def __init__(self, num_records=5000):
        self.num_records = num_records
        self.load_dataset_stats()
        self.current_time = datetime(2004, 3, 1, 0, 0, 0)  # Start time
        self.moteid = 1
        
        # Trend control variables
        self.trend_duration = 0
        self.trend_direction = 'stable'  # 'increase', 'decrease', 'stable'
        self.trend_target = {}
        self.anomaly_duration = 0
        self.anomaly_active = False
        
        # Current sensor values
        self.current_values = {
            'temperature': 22.0,
            'humidity': 40.0,
            'light': 150.0,
            'voltage': 2.5
        }
        
        # Trend parameters
        self.trend_change_rate = 0.02  # How much to change per step
        self.anomaly_probability = 0.05  # 5% chance of anomaly
        
    def load_dataset_stats(self):
        """Load dataset statistics for realistic simulation"""
        try:
            stats_path = os.path.join('models', 'dataset_stats.joblib')
            if os.path.exists(stats_path):
                self.stats = joblib.load(stats_path)
                print("✅ Loaded dataset statistics")
            else:
                # Default stats based on your dataset
                self.stats = {
                    'temperature': {'mean': 22.1, 'std': 3.3, 'min': -38.4, 'max': 385.6},
                    'humidity': {'mean': 39.3, 'std': 6.2, 'min': -4.0, 'max': 137.5},
                    'light': {'mean': 415.4, 'std': 544.3, 'min': 0.0, 'max': 1847.4},
                    'voltage': {'mean': 2.56, 'std': 0.11, 'min': 2.27, 'max': 3.16}
                }
                print("⚠️ Using default statistics")
        except Exception as e:
            print(f"❌ Error loading stats: {e}")
            # Use default values
            self.stats = {
                'temperature': {'mean': 22.1, 'std': 3.3, 'min': -38.4, 'max': 385.6},
                'humidity': {'mean': 39.3, 'std': 6.2, 'min': -4.0, 'max': 137.5},
                'light': {'mean': 415.4, 'std': 544.3, 'min': 0.0, 'max': 1847.4},
                'voltage': {'mean': 2.56, 'std': 0.11, 'min': 2.27, 'max': 3.16}
            }
    
    def decide_trend_change(self):
        """Decide if trend should change and set new trend parameters"""
        if self.trend_duration <= 0:
            # Start new trend
            self.trend_duration = random.randint(50, 200)  # Trend lasts 50-200 records
            
            trend_options = ['increase', 'decrease', 'stable']
            self.trend_direction = random.choice(trend_options)
            
            # Set trend targets for parameters
            for param in ['temperature', 'humidity', 'light', 'voltage']:
                if self.trend_direction == 'increase':
                    self.trend_target[param] = self.current_values[param] + random.uniform(1.0, 5.0)
                elif self.trend_direction == 'decrease':
                    self.trend_target[param] = self.current_values[param] - random.uniform(1.0, 5.0)
                else:  # stable
                    self.trend_target[param] = self.current_values[param]
                
                # Keep within realistic bounds
                stats = self.stats[param]
                self.trend_target[param] = max(min(self.trend_target[param], 
                                                stats['max']), stats['min'])
        
        self.trend_duration -= 1
    
    def decide_anomaly_state(self):
        """Decide if sensor should be in anomaly state"""
        if not self.anomaly_active:
            # Check if anomaly should start
            if random.random() < self.anomaly_probability:
                self.anomaly_active = True
                self.anomaly_duration = random.randint(10, 50)  # Anomaly lasts 10-50 records
                print(f"🚨 Anomaly started - Duration: {self.anomaly_duration}")
        else:
            # Continue anomaly
            self.anomaly_duration -= 1
            if self.anomaly_duration <= 0:
                self.anomaly_active = False
                print("✅ Anomaly ended")
    
    def generate_normal_value(self, param):
        """Generate normal sensor value with smooth trends"""
        current = self.current_values[param]
        target = self.trend_target.get(param, current)
        
        # Move slowly towards target
        if abs(current - target) > 0.1:
            change = (target - current) * self.trend_change_rate
        else:
            change = 0
        
        # Add small random variation
        noise = random.uniform(-0.1, 0.1)
        new_value = current + change + noise
        
        # Keep within realistic bounds
        stats = self.stats[param]
        new_value = max(min(new_value, stats['max']), stats['min'])
        
        return new_value
    
    def generate_anomaly_value(self, param):
        """Generate anomalous sensor value"""
        current = self.current_values[param]
        stats = self.stats[param]
        
        # Create anomaly based on parameter type
        if param == 'temperature':
            # Temperature spikes or drops
            if random.random() < 0.5:
                return current + random.uniform(50, 100)  # High temperature
            else:
                return max(current - random.uniform(20, 40), stats['min'])  # Low temperature
        
        elif param == 'humidity':
            # Humidity extremes
            if random.random() < 0.5:
                return min(current + random.uniform(30, 50), stats['max'])  # High humidity
            else:
                return max(current - random.uniform(30, 50), stats['min'])  # Low humidity
        
        elif param == 'light':
            # Light sensor failures or very high values
            if random.random() < 0.3:
                return 0  # Sensor failure
            else:
                return min(current * random.uniform(2, 5), stats['max'])  # Very bright
        
        elif param == 'voltage':
            # Voltage drops (common in sensor anomalies)
            return max(current - random.uniform(0.3, 0.8), stats['min'])
        
        return current
    
    def generate_single_record(self):
        """Generate a single IoT sensor record"""
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
            'status': ''  # Initially empty as requested
        }
        
        # Increment time (30 seconds to 2 minutes interval)
        time_increment = random.randint(30, 120)
        self.current_time += timedelta(seconds=time_increment)
        
        return record
    
    def generate_dataset(self):
        """Generate complete IoT dataset"""
        print(f"🔄 Generating {self.num_records} IoT sensor records...")
        
        records = []
        for i in range(self.num_records):
            record = self.generate_single_record()
            records.append(record)
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"   Generated {i + 1}/{self.num_records} records...")
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Save to intermediate.csv
        df.to_csv('intermediate.csv', index=False)
        print(f"✅ Generated {len(df)} records saved to intermediate.csv")
        
        # Display statistics
        print("\n📊 Generated Dataset Statistics:")
        print(f"Temperature range: {df['temperature'].min():.2f} to {df['temperature'].max():.2f}")
        print(f"Humidity range: {df['humidity'].min():.2f} to {df['humidity'].max():.2f}")
        print(f"Light range: {df['light'].min():.2f} to {df['light'].max():.2f}")
        print(f"Voltage range: {df['voltage'].min():.5f} to {df['voltage'].max():.5f}")
        print(f"Time span: {df['date'].min()} to {df['date'].max()}")
        
        return df

def main():
    """Main simulation function"""
    print("🚀 IoT Sensor Data Simulator")
    print("="*40)
    
    # Get number of records to generate
    try:
        num_records = int(input("Enter number of records to generate (default 5000): ") or 5000)
    except ValueError:
        num_records = 5000
        print("Using default: 5000 records")
    
    # Create simulator and generate data
    simulator = IoTSimulator(num_records)
    dataset = simulator.generate_dataset()
    
    print("\n✅ Simulation completed successfully!")
    print("📁 Data saved to: intermediate.csv")
    print("🔄 Next step: Run preprocessing and model inference")

if __name__ == "__main__":
    main()