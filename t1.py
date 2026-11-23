import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def generate_total_vehicle_data(samples=1000):
    np.random.seed(42)
    data = {
        'odo': np.random.uniform(0, 150000, samples),
        'torque_aggression': np.random.uniform(0.5, 4.0, samples), # Propulsion stress
        'road_vibration': np.random.uniform(1, 10, samples),      # Structural stress
        'temp_avg': np.random.uniform(-10, 45, samples),         # Chemical stress
        'payload_kg': np.random.uniform(0, 600, samples)          # Static stress
    }
    df = pd.DataFrame(data)

    # TARGETS: The AI learns these wear patterns
    # Moving Parts (Drivetrain/Bearings) - Odo * Torque^1.8
    df['moving_wear'] = (df['odo'] / 1800) * (df['torque_aggression']**1.8)
    
    # Holding Parts (Bushings/Chassis) - Odo * Vibration * Weight
    df['holding_wear'] = (df['odo'] / 1200) * (df['road_vibration'] * 0.4) * (1 + df['payload_kg']/400)
    
    # Electronic Parts (Inverter/DC-DC) - Odo * Temp * Current
    df['elec_wear'] = (df['odo'] / 2500) * (1.5 if df['temp_avg'].mean() > 30 else 1.0)

    return df.clip(0, 100)

training_df = generate_total_vehicle_data()