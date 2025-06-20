import time
import csv
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

class DataLogger:
    def __init__(self, base_name, record_duration=20):
        self.base_name = base_name
        self.record_duration = record_duration
        self.csv_file = None
        self.csv_writer = None
        self.start_time = None
        self.odom_data = None
        self.low_data = None
        self.open_new_csv_file()

    def open_new_csv_file(self):
        if self.csv_file:
            self.csv_file.close()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.base_name}_{timestamp}.csv"
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        columns = [
            'timestamp',
            'odom_stamp_sec', 'odom_stamp_nanosec',
            'odom_mode',
            'odom_imu_quaternion_w', 'odom_imu_quaternion_x', 'odom_imu_quaternion_y', 'odom_imu_quaternion_z',
            'odom_imu_angular_velocity_x', 'odom_imu_angular_velocity_y', 'odom_imu_angular_velocity_z',
            'odom_imu_linear_acceleration_x', 'odom_imu_linear_acceleration_y', 'odom_imu_linear_acceleration_z',
            'odom_imu_temperature',
            'odom_gait_type',
            'odom_position_mode',
            'odom_velocity_mode',
            'odom_yaw',
            'odom_position_x', 'odom_position_y', 'odom_position_z',
            'odom_yaw_speed',
            'odom_velocity_x', 'odom_velocity_y', 'odom_velocity_z',
            'odom_angular_speed',
            'odom_foot_position_1', 'odom_foot_position_2', 'odom_foot_position_3', 'odom_foot_position_4',
            'odom_foot_contact_1', 'odom_foot_contact_2', 'odom_foot_contact_3', 'odom_foot_contact_4',
            'odom_foot_force_1', 'odom_foot_force_2', 'odom_foot_force_3', 'odom_foot_force_4',
            'odom_foot_force_5', 'odom_foot_force_6', 'odom_foot_force_7', 'odom_foot_force_8',
            'odom_foot_force_9', 'odom_foot_force_10', 'odom_foot_force_11', 'odom_foot_force_12',
            'odom_foot_position_x1', 'odom_foot_position_y1', 'odom_foot_position_z1',
            'odom_foot_position_x2', 'odom_foot_position_y2', 'odom_foot_position_z2',
            'odom_foot_position_x3', 'odom_foot_position_y3', 'odom_foot_position_z3',
            'odom_foot_position_x4', 'odom_foot_position_y4', 'odom_foot_position_z4',
            *[f'odom_path_point_{i+1}_{field}' for i in range(10) for field in ['x', 'y', 'yaw', 'vx', 'vy', 'time']],
            'low_tick', 'low_version_0', 'low_version_1',
            'low_mode_pr', 'low_mode_machine',
            'low_imu_quat_w', 'low_imu_quat_x', 'low_imu_quat_y', 'low_imu_quat_z',
            'low_imu_gyro_x', 'low_imu_gyro_y', 'low_imu_gyro_z',
            'low_imu_accel_x', 'low_imu_accel_y', 'low_imu_accel_z',
            'low_imu_roll', 'low_imu_pitch', 'low_imu_yaw', 'low_imu_temperature'
        ]
        for i in range(35):
            columns += [
                f'low_motor_{i}_mode', f'low_motor_{i}_q', f'low_motor_{i}_dq',
                f'low_motor_{i}_ddq', f'low_motor_{i}_tau_est',
                f'low_motor_{i}_temp_0', f'low_motor_{i}_temp_1',
                f'low_motor_{i}_sensor_0', f'low_motor_{i}_sensor_1',
                f'low_motor_{i}_vol', f'low_motor_{i}_motorstate'
            ] + [f'low_motor_{i}_reserve_{j}' for j in range(4)]
        columns += [f'low_wireless_remote_{i}' for i in range(40)]
        columns += [f'low_reserve_{i}' for i in range(4)]
        columns += ['low_crc']
        self.csv_writer.writerow(columns)
        self.start_time = time.time()

    def odom_callback(self, msg):
        current_time = time.time()
        if current_time - self.start_time >= self.record_duration:
            self.open_new_csv_file()
        try:
            stamp = getattr(msg, 'stamp', None)
            sec = stamp.sec if stamp else 0
            nanosec = stamp.nanosec if stamp else 0
            position_mode = getattr(msg, 'position_mode', 0)
            velocity_mode = getattr(msg, 'velocity_mode', 0)
            angular_speed = getattr(msg, 'angular_speed', 0.0)
            foot_contact = msg.foot_force[:4] if len(msg.foot_force) >= 4 else [0] * 4
            foot_force = msg.foot_force + [0] * (12 - len(msg.foot_force)) if len(msg.foot_force) < 12 else msg.foot_force[:12]
            foot_position_body = msg.foot_position_body + [0] * (12 - len(msg.foot_position_body)) if len(msg.foot_position_body) < 12 else msg.foot_position_body[:12]

            odom_row = [
                sec, nanosec,
                msg.mode,
                msg.imu_state.quaternion[0], msg.imu_state.quaternion[1], msg.imu_state.quaternion[2], msg.imu_state.quaternion[3],
                msg.imu_state.gyroscope[0], msg.imu_state.gyroscope[1], msg.imu_state.gyroscope[2],
                msg.imu_state.accelerometer[0], msg.imu_state.accelerometer[1], msg.imu_state.accelerometer[2],
                msg.imu_state.temperature,
                msg.gait_type,
                position_mode,
                velocity_mode,
                msg.imu_state.rpy[2],
                msg.position[0], msg.position[1], msg.position[2],
                msg.yaw_speed,
                msg.velocity[0], msg.velocity[1], msg.velocity[2],
                angular_speed,
                foot_position_body[0], foot_position_body[1], foot_position_body[2], foot_position_body[3],
                foot_contact[0], foot_contact[1], foot_contact[2], foot_contact[3],
                foot_force[0], foot_force[1], foot_force[2], foot_force[3], foot_force[4], foot_force[5],
                foot_force[6], foot_force[7], foot_force[8], foot_force[9], foot_force[10], foot_force[11],
                foot_position_body[0], foot_position_body[1], foot_position_body[2],
                foot_position_body[3], foot_position_body[4], foot_position_body[5],
                foot_position_body[6], foot_position_body[7], foot_position_body[8],
                foot_position_body[9], foot_position_body[10], foot_position_body[11],
                *[field for i in range(10) for field in [
                    msg.path_point[i].x, msg.path_point[i].y,
                    msg.path_point[i].yaw, msg.path_point[i].vx, msg.path_point[i].vy, msg.path_point[i].t_from_start
                ]]
            ]
            self.odom_data = odom_row
            self.write_row(current_time)
        except AttributeError as e:
            print(f"Error processing odom message: {e}")
        except IndexError as e:
            print(f"Index error in odom message: {e}")

    def low_callback(self, msg):
        current_time = time.time()
        if current_time - self.start_time >= self.record_duration:
            self.open_new_csv_file()
        try:
            low_row = [
                msg.tick, msg.version[0], msg.version[1],
                msg.mode_pr, msg.mode_machine
            ]
            imu = msg.imu_state
            low_row += [
                imu.quaternion[0], imu.quaternion[1], imu.quaternion[2], imu.quaternion[3],
                imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2],
                imu.accelerometer[0], imu.accelerometer[1], imu.accelerometer[2],
                imu.rpy[0], imu.rpy[1], imu.rpy[2],
                imu.temperature
            ]
            for motor in msg.motor_state[:35]:
                low_row += [
                    motor.mode, motor.q, motor.dq, motor.ddq, motor.tau_est,
                    motor.temperature[0], motor.temperature[1],
                    motor.sensor[0], motor.sensor[1],
                    motor.vol, motor.motorstate
                ] + list(motor.reserve)
            for _ in range(35 - len(msg.motor_state)):
                low_row += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
            low_row += list(msg.wireless_remote)
            low_row += list(msg.reserve)
            low_row += [msg.crc]
            self.low_data = low_row
            self.write_row(current_time)
        except AttributeError as e:
            print(f"Error processing low message: {e}")
        except IndexError as e:
            print(f"Index error in low message: {e}")

    def write_row(self, current_time):
        odom_row = self.odom_data if self.odom_data else [0] * (2 + 1 + 4 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 3 + 1 + 3 + 1 + 4 + 4 + 12 + 12 + 10 * 6)
        low_row = self.low_data if self.low_data else [0] * (1 + 2 + 2 + 4 + 3 + 3 + 1 + 35 * (11 + 4) + 40 + 4 + 1)
        row = [current_time] + odom_row + low_row
        self.csv_writer.writerow(row)
        if int(current_time * 1000) % 100 == 0:
            self.csv_file.flush()

    def run(self):
        ChannelFactoryInitialize(0)
        # 移除 qos_depth，假设不支持此参数
        odom_sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
        low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        try:
            odom_sub.Init(self.odom_callback, 10)  # 使用默认队列深度
            low_sub.Init(self.low_callback, 10)
            while True:
                time.sleep(1)
        except Exception as e:
            print(f"Error in subscriber: {e}")
        finally:
            self.__del__()

    def __del__(self):
        if self.csv_file:
            self.csv_file.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python data_logger.py <base_name>")
        sys.exit(1)
    logger = DataLogger(sys.argv[1])
    logger.run()