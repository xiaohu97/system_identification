import sys
import pandas as pd
import numpy as np

def main():
    # 检查是否提供了 CSV 文件路径
    if len(sys.argv) < 2:
        print("用法: python script.py <csv_file_path>")
        sys.exit(1)

    # 从命令行参数获取 CSV 文件路径
    csv_file_path = sys.argv[1]

    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 定义每个 .dat 文件的列列表
    low_q_cols = [
        'odom_position_x', 'odom_position_y', 'odom_position_z',
        'low_imu_quat_x', 'low_imu_quat_y', 'low_imu_quat_z', 'low_imu_quat_w'
    ] + [f'low_motor_{i}_q' for i in range(12)]

    odom_q_cols = [
        'odom_position_x', 'odom_position_y', 'odom_position_z',
        'odom_imu_quaternion_x', 'odom_imu_quaternion_y', 'odom_imu_quaternion_z', 'odom_imu_quaternion_w'
    ] + [f'low_motor_{i}_q' for i in range(12)]

    dq_cols = [
        'odom_velocity_x', 'odom_velocity_y', 'odom_velocity_z',
        'low_imu_gyro_x', 'low_imu_gyro_y', 'low_imu_gyro_z'
    ] + [f'low_motor_{i}_dq' for i in range(12)]

    ddq_cols = [
        'low_imu_accel_x', 'low_imu_accel_y', 'low_imu_accel_z',
        'body_ang_acceleration_x', 'body_ang_acceleration_y', 'body_ang_acceleration_z'
    ] + [f'low_motor_{i}_ddq' for i in range(1, 12)]

    tau_cols = [f'low_motor_{i}_tau_est' for i in range(12)]

    contact_cols = ['odom_foot_contact_1', 'odom_foot_contact_2']

    # 检查缺失的列
    required_cols = set(low_q_cols + odom_q_cols + dq_cols + ddq_cols + tau_cols + contact_cols)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"CSV 中缺少列: {missing_cols}")
        sys.exit(1)

    # 提取并保存数据到 .dat 文件
    np.savetxt('g1_robot_low_q.dat', df[low_q_cols].to_numpy().T, delimiter='\t', fmt='%.6f')
    np.savetxt('g1_robot_odom_q.dat', df[odom_q_cols].to_numpy().T, delimiter='\t', fmt='%.6f')
    np.savetxt('g1_robot_dq.dat', df[dq_cols].to_numpy().T, delimiter='\t', fmt='%.6f')
    np.savetxt('g1_robot_ddq.dat', df[ddq_cols].to_numpy().T, delimiter='\t', fmt='%.6f')
    np.savetxt('g1_robot_tau.dat', df[tau_cols].to_numpy().T, delimiter='\t', fmt='%.6f')
    np.savetxt('g1_robot_contact.dat', df[contact_cols].to_numpy().T, delimiter='\t', fmt='%.6f')

    print("提取的数据已保存到 .dat 文件。")

if __name__ == "__main__":
    main()