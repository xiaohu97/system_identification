import time
import csv
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

class DataLogger:
    def __init__(self, odom_base_name, low_base_name, record_duration=20):
        self.odom_base_name = odom_base_name
        self.low_base_name = low_base_name
        self.record_duration = record_duration  # 记录时长（秒）

        # 初始化文件和写入器
        self.odom_csv = None
        self.low_csv = None
        self.odom_writer = None
        self.low_writer = None
        self.odom_start_time = None
        self.low_start_time = None

        # 打开初始文件
        self.open_new_odom_file()
        self.open_new_low_file()

    def open_new_odom_file(self):
        if self.odom_csv:
            self.odom_csv.close()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.odom_base_name}_{timestamp}.csv"
        self.odom_csv = open(filename, 'w', newline='')
        self.odom_writer = csv.writer(self.odom_csv)
        self.odom_writer.writerow([
            'timestamp', 'pos_x', 'pos_y', 'pos_z', 
            'vel_x', 'vel_y', 'vel_z', 
            'roll', 'pitch', 'yaw', 'yaw_speed', 
            'quat_w', 'quat_x', 'quat_y', 'quat_z'
        ])
        self.odom_start_time = time.time()

    def open_new_low_file(self):
        if self.low_csv:
            self.low_csv.close()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.low_base_name}_{timestamp}.csv"
        self.low_csv = open(filename, 'w', newline='')
        self.low_writer = csv.writer(self.low_csv)
        low_columns = [
            'timestamp', 'tick', 'version_0', 'version_1', 
            'mode_pr', 'mode_machine',
            'imu_quat_w', 'imu_quat_x', 'imu_quat_y', 'imu_quat_z',
            'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z',
            'imu_accel_x', 'imu_accel_y', 'imu_accel_z',
            'imu_roll', 'imu_pitch', 'imu_yaw', 'imu_temperature'
        ]
        for i in range(35):
            low_columns += [
                f'motor_{i}_mode', f'motor_{i}_q', f'motor_{i}_dq', 
                f'motor_{i}_ddq', f'motor_{i}_tau_est', 
                f'motor_{i}_q_raw', f'motor_{i}_dq_raw', f'motor_{i}_ddq_raw',
                f'motor_{i}_temp_0', f'motor_{i}_temp_1',
                f'motor_{i}_sensor_0', f'motor_{i}_sensor_1',
                f'motor_{i}_vol', f'motor_{i}_motorstate'
            ] + [f'motor_{i}_reserve_{j}' for j in range(4)]
        low_columns += [f'wireless_remote_{i}' for i in range(40)]
        low_columns += [f'reserve_{i}' for i in range(4)]
        low_columns += ['crc']
        self.low_writer.writerow(low_columns)
        self.low_start_time = time.time()

    def odom_callback(self, msg):
        current_time = time.time()
        # 检查是否需要创建新文件
        if current_time - self.odom_start_time >= self.record_duration:
            self.open_new_odom_file()

        row = [
            current_time,
            msg.position()[0], msg.position()[1], msg.position()[2],
            msg.velocity()[0], msg.velocity()[1], msg.velocity()[2],
            msg.imu_state().rpy()[0], msg.imu_state().rpy()[1], msg.imu_state().rpy()[2],
            msg.yaw_speed(),
            msg.imu_state().quaternion()[0], msg.imu_state().quaternion()[1],
            msg.imu_state().quaternion()[2], msg.imu_state().quaternion()[3]
        ]
        self.odom_writer.writerow(row)
        self.odom_csv.flush()

    def low_callback(self, msg):
        current_time = time.time()
        # 检查是否需要创建新文件
        if current_time - self.low_start_time >= self.record_duration:
            self.open_new_low_file()

        row = [
            current_time, msg.tick, msg.version[0], msg.version[1],
            msg.mode_pr, msg.mode_machine
        ]
        imu = msg.imu_state
        row += [
            imu.quaternion[0], imu.quaternion[1], imu.quaternion[2], imu.quaternion[3],
            imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2],
            imu.accelerometer[0], imu.accelerometer[1], imu.accelerometer[2],
            imu.rpy[0], imu.rpy[1], imu.rpy[2],
            imu.temperature
        ]
        for motor in msg.motor_state:
            row += [
                motor.mode, motor.q, motor.dq, motor.ddq, motor.tau_est,
                motor.q_raw, motor.dq_raw, motor.ddq_raw,
                motor.temperature[0], motor.temperature[1],
                motor.sensor[0], motor.sensor[1],
                motor.vol, motor.motorstate
            ] + list(motor.reserve)
        row += list(msg.wireless_remote)
        row += list(msg.reserve)
        row += [msg.crc]
        self.low_writer.writerow(row)
        self.low_csv.flush()

    def run(self):
        ChannelFactoryInitialize(0)
        odom_sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
        odom_sub.Init(self.odom_callback, 10)
        low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        low_sub.Init(self.low_callback, 10)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("程序已终止")

    def __del__(self):
        if self.odom_csv:
            self.odom_csv.close()
        if self.low_csv:
            self.low_csv.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python data_logger.py <odom_base_name> <low_base_name>")
        sys.exit(1)
    logger = DataLogger(sys.argv[1], sys.argv[2])
    logger.run()
    
    
