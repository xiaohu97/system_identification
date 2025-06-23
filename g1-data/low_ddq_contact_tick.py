import sys
import pandas as pd
import numpy as np
import os

def calculate_low_motor_ddq(csv_file_path, motor_count=35):
    """
    Calculate low_motor_{i}_ddq, update odom_foot_contact_1 and odom_foot_contact_2,
    and compute body angular accelerations based on IMU gyroscope data.

    Args:
        csv_file_path (str): Path to the input CSV file.
        motor_count (int): Number of motors (default 35).
    """
    # Check if file exists
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Define required columns
    timestamp_col = 'low_tick'
    dq_cols = [f'low_motor_{i}_dq' for i in range(motor_count)]
    tau_cols = ['low_motor_4_tau_est', 'low_motor_10_tau_est']
    contact_cols = ['odom_foot_contact_1', 'odom_foot_contact_2']
    gyro_cols = ['low_imu_gyro_x', 'low_imu_gyro_y', 'low_imu_gyro_z']
    required_cols = [timestamp_col] + dq_cols + tau_cols + contact_cols + gyro_cols

    # Verify required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    # Initialize ddq and body angular acceleration columns
    for i in range(motor_count):
        df[f'low_motor_{i}_ddq'] = np.nan
    df['body_ang_acceleration_x'] = np.nan
    df['body_ang_acceleration_y'] = np.nan
    df['body_ang_acceleration_z'] = np.nan

    # Calculate ddq and body angular accelerations from second row onward
    for row in range(1, len(df)):
        delta_time = df.at[row, timestamp_col] - df.at[row - 1, timestamp_col]
        # Calculate ddq for each motor
        for i in range(motor_count):
            dq_col = f'low_motor_{i}_dq'
            ddq_col = f'low_motor_{i}_ddq'
            delta_dq = df.at[row, dq_col] - df.at[row - 1, dq_col]
            if delta_time > 0:
                df.at[row, ddq_col] = delta_dq * 1000 / delta_time  # Convert to seconds
            elif delta_dq == 0:
                df.at[row, ddq_col] = 0.0
            else:
                df.at[row, ddq_col] = np.nan
        # Calculate body angular accelerations
        for axis in ['x', 'y', 'z']:
            gyro_col = f'low_imu_gyro_{axis}'
            accel_col = f'body_ang_acceleration_{axis}'
            delta_gyro = df.at[row, gyro_col] - df.at[row - 1, gyro_col]
            if delta_time > 0:
                df.at[row, accel_col] = delta_gyro * 1000 / delta_time
            elif delta_gyro == 0:
                df.at[row, accel_col] = 0.0
            else:
                df.at[row, accel_col] = np.nan

    # Update odom_foot_contact_1 based on low_motor_4_tau_est
    df['odom_foot_contact_1'] = np.where(
        df['low_motor_4_tau_est'] >= 10, 1,
        np.where(df['low_motor_4_tau_est'] > -5, 2, 0)
    )

    # Update odom_foot_contact_2 based on low_motor_10_tau_est
    df['odom_foot_contact_2'] = np.where(
        df['low_motor_10_tau_est'] >= 10, 1,
        np.where(df['low_motor_10_tau_est'] > -5, 2, 0)
    )

    # Save to new CSV
    updated_csv_file_path = csv_file_path.replace('.csv', '_updated_tick.csv')
    try:
        df.to_csv(updated_csv_file_path, index=False)
        print(f"更新后的 CSV 文件已保存至： {updated_csv_file_path}")
    except Exception as e:
        raise ValueError(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    # Check if a CSV file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python calculate_low_motor_ddq.py <csv_file_path>")
        sys.exit(1)

    # Get CSV file path from command-line argument
    csv_file_path = sys.argv[1]

    # Verify file existence
    if not os.path.isfile(csv_file_path):
        print(f"Error: CSV file not found - {csv_file_path}")
        sys.exit(1)

    # Run the calculation
    calculate_low_motor_ddq(csv_file_path, motor_count=35)