import pandas as pd
import numpy as np

def calculate_low_motor_ddq(csv_file_path, motor_count=35):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 提取 timestamp 列
    timestamp_col = 'timestamp'

    # 提取 low_motor_{i}_dq 列
    dq_cols = [f'low_motor_{i}_dq' for i in range(motor_count)]

    # 为 low_motor_{i}_ddq 创建新列并初始化为 NaN
    for i in range(motor_count):
        df[f'low_motor_{i}_ddq'] = np.nan

    # 从第二行开始计算 ddq
    for row in range(1, len(df)):
        delta_time = df.at[row, timestamp_col] - df.at[row - 1, timestamp_col]
        for i in range(motor_count):
            dq_col = f'low_motor_{i}_dq'
            ddq_col = f'low_motor_{i}_ddq'
            delta_dq = df.at[row, dq_col] - df.at[row - 1, dq_col]
            
            # 应用计算逻辑
            if delta_time > 0:
                df.at[row, ddq_col] = delta_dq / delta_time
            elif delta_dq == 0:
                df.at[row, ddq_col] = 0.0
            else:
                df.at[row, ddq_col] = np.nan

    # 保存更新后的 CSV 文件
    updated_csv_file_path = csv_file_path.replace('.csv', '_updated.csv')
    df.to_csv(updated_csv_file_path, index=False)
    print(f"更新后的 CSV 文件已保存至：{updated_csv_file_path}")

# 使用示例
if __name__ == "__main__":
    csv_file_path = 'output_g1_1_20250620_170633.csv'  # 替换为您的 CSV 文件路径
    calculate_low_motor_ddq(csv_file_path)