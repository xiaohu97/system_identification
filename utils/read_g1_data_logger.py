import time
import csv
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# timestamp [s] and [10^{-9}s]
    # - seconds
    # - nanoseconds    

    # position [m] 
    # - base:   body_lin_x	body_lin_y	body_lin_z	四元数 body_ang_x	body_ang_y	body_ang_z	body_ang_w
    # - joints: fl.hx       fl.hy	    fl.kn   	fr.hx	    fr.hy	    fr.kn   	
    #           hl.hx	    hl.hy	    hl.kn	    hr.hx	    hr.hy	    hr.kn

    # velocity [m/s]
    # - base:   body_lin_x	body_lin_y	body_lin_z	body_ang_x	body_ang_y	body_ang_z
    # - joints: fl.hx       fl.hy	    fl.kn   	fr.hx	    fr.hy	    fr.kn   	
    #           hl.hx	    hl.hy	    hl.kn	    hr.hx	    hr.hy	    hr.kn

    # acceleration [m/s]
    # - base:   body_lin_x	body_lin_y	body_lin_z	body_ang_x	body_ang_y	body_ang_z
    # - joints: fl.hx       fl.hy	    fl.kn   	fr.hx	    fr.hy	    fr.kn   	
    #           hl.hx	    hl.hy	    hl.kn	    hr.hx	    hr.hy	    hr.kn

    # loads [Nm]
    # - joints: fl.hx       fl.hy	    fl.kn   	fr.hx	    fr.hy	    fr.kn   	
    #           hl.hx	    hl.hy	    hl.kn	    hr.hx	    hr.hy	    hr.kn

    # foot_state []
    # - foot in contact: CONTACT_UNKNOWN=0, CONTACT_MADE=1, CONTACT_LOST=2

    # the base position and velocity can be measured in a odom or vison frame
    # the base acceleration can not be measured yet because the RobotStateStreamingService, which is needed to read the IMU data, is still in beta.


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
            'timestamp_sec', 'timestamp_nanosec',  # 时间戳（TimeSpec_）
            'mode',                                # 整数（模式/状态）
            'imu_quaternion_w', 'imu_quaternion_x', 'imu_quaternion_y', 'imu_quaternion_z',  # IMU 四元数
            'imu_angular_velocity_x', 'imu_angular_velocity_y', 'imu_angular_velocity_z',    # IMU 角速度
            'imu_linear_acceleration_x', 'imu_linear_acceleration_y', 'imu_linear_acceleration_z',  # IMU 线性加速度
            'imu_temperature',                     # IMU 温度
            'gait_type',                           # 整数（步态类型）
            'position_mode',                       # 整数（位置模式）
            'velocity_mode',                       # 整数（速度模式）
            'yaw',                                 # 浮点数（偏航角）
            'position_x', 'position_y', 'position_z',  # 浮点数数组（位置）
            'yaw_speed',                           # 浮点数（偏航角速度）
            'velocity_x', 'velocity_y', 'velocity_z',  # 浮点数数组（速度）
            'angular_speed',                       # 浮点数（角速度）
            'foot_position_1', 'foot_position_2', 'foot_position_3', 'foot_position_4',  # 浮点数数组（足部位置/偏移）
            'foot_contact_1', 'foot_contact_2', 'foot_contact_3', 'foot_contact_4',      # 整数数组（足部接触状态）
            'foot_force_1', 'foot_force_2', 'foot_force_3', 'foot_force_4', 'foot_force_5', 'foot_force_6', 
            'foot_force_7', 'foot_force_8', 'foot_force_9', 'foot_force_10', 'foot_force_11', 'foot_force_12',  # 浮点数数组（足部力）
            'foot_position_x1', 'foot_position_y1', 'foot_position_z1', 
            'foot_position_x2', 'foot_position_y2', 'foot_position_z2', 
            'foot_position_x3', 'foot_position_y3', 'foot_position_z3', 
            'foot_position_x4', 'foot_position_y4', 'foot_position_z4',  # 浮点数数组（足部位置）
            'path_point_1_x', 'path_point_1_y', 'path_point_1_z', 'path_point_1_yaw', 
            'path_point_1_vx', 'path_point_1_vy', 'path_point_1_time',  # 路径点 1
            'path_point_2_x', 'path_point_2_y', 'path_point_2_z', 'path_point_2_yaw', 
            'path_point_2_vx', 'path_point_2_vy', 'path_point_2_time',  # 路径点 2
            'path_point_3_x', 'path_point_3_y', 'path_point_3_z', 'path_point_3_yaw', 
            'path_point_3_vx', 'path_point_3_vy', 'path_point_3_time',  # 路径点 3
            'path_point_4_x', 'path_point_4_y', 'path_point_4_z', 'path_point_4_yaw', 
            'path_point_4_vx', 'path_point_4_vy', 'path_point_4_time',  # 路径点 4
            'path_point_5_x', 'path_point_5_y', 'path_point_5_z', 'path_point_5_yaw', 
            'path_point_5_vx', 'path_point_5_vy', 'path_point_5_time',  # 路径点 5
            'path_point_6_x', 'path_point_6_y', 'path_point_6_z', 'path_point_6_yaw', 
            'path_point_6_vx', 'path_point_6_vy', 'path_point_6_time',  # 路径点 6
            'path_point_7_x', 'path_point_7_y', 'path_point_7_z', 'path_point_7_yaw', 
            'path_point_7_vx', 'path_point_7_vy', 'path_point_7_time',  # 路径点 7
            'path_point_8_x', 'path_point_8_y', 'path_point_8_z', 'path_point_8_yaw', 
            'path_point_8_vx', 'path_point_8_vy', 'path_point_8_time',  # 路径点 8
            'path_point_9_x', 'path_point_9_y', 'path_point_9_z', 'path_point_9_yaw', 
            'path_point_9_vx', 'path_point_9_vy', 'path_point_9_time',  # 路径点 9
            'path_point_10_x', 'path_point_10_y', 'path_point_10_z', 'path_point_10_yaw', 
            'path_point_10_vx', 'path_point_10_vy', 'path_point_10_time'  # 路径点 10
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
            msg.timespec.sec, msg.timespec.nanosec,  # TimeSpec_
            msg.mode,                                # mode
            msg.imu_state.quaternion[0], msg.imu_state.quaternion[1], msg.imu_state.quaternion[2], msg.imu_state.quaternion[3],  # IMU quaternion
            msg.imu_state.angular_velocity[0], msg.imu_state.angular_velocity[1], msg.imu_state.angular_velocity[2],  # IMU angular velocity
            msg.imu_state.linear_acceleration[0], msg.imu_state.linear_acceleration[1], msg.imu_state.linear_acceleration[2],  # IMU linear acceleration
            msg.imu_state.temperature,               # IMU temperature
            msg.gait_type,                           # gait type
            msg.position_mode,                       # position mode
            msg.velocity_mode,                       # velocity mode
            msg.yaw,                                 # yaw
            msg.position[0], msg.position[1], msg.position[2],  # position
            msg.yaw_speed,                           # yaw speed
            msg.velocity[0], msg.velocity[1], msg.velocity[2],  # velocity
            msg.angular_speed,                       # angular speed
            msg.foot_position[0], msg.foot_position[1], msg.foot_position[2], msg.foot_position[3],  # foot positions
            msg.foot_contact[0], msg.foot_contact[1], msg.foot_contact[2], msg.foot_contact[3],  # foot contact states
            msg.foot_force[0], msg.foot_force[1], msg.foot_force[2], msg.foot_force[3], msg.foot_force[4], msg.foot_force[5],
            msg.foot_force[6], msg.foot_force[7], msg.foot_force[8], msg.foot_force[9], msg.foot_force[10], msg.foot_force[11],  # foot forces
            msg.foot_position_global[0], msg.foot_position_global[1], msg.foot_position_global[2],  # foot position 1
            msg.foot_position_global[3], msg.foot_position_global[4], msg.foot_position_global[5],  # foot position 2
            msg.foot_position_global[6], msg.foot_position_global[7], msg.foot_position_global[8],  # foot position 3
            msg.foot_position_global[9], msg.foot_position_global[10], msg.foot_position_global[11],  # foot position 4
            # Path points (10 points, each with 7 fields: x, y, z, yaw, vx, vy, time)
            *[field for i in range(10) for field in [
                msg.path_point[i].x, msg.path_point[i].y, msg.path_point[i].z,
                msg.path_point[i].yaw, msg.path_point[i].vx, msg.path_point[i].vy, msg.path_point[i].time
            ]]
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
    
    
