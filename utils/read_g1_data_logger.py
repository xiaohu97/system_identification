import time
import csv
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

class DataLogger:
    def __init__(self, odom_base_name, low_base_name, record_duration=20):
        self.odom_base_name = odom_base_name
        self.low_base_name = low_base_name
        self.record_duration = record_duration
        self.odom_csv = None
        self.low_csv = None
        self.odom_writer = None
        self.low_writer = None
        self.odom_start_time = None
        self.low_start_time = None
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
            'timestamp_sec', 'timestamp_nanosec',
            'mode',
            'imu_quaternion_w', 'imu_quaternion_x', 'imu_quaternion_y', 'imu_quaternion_z',
            'imu_angular_velocity_x', 'imu_angular_velocity_y', 'imu_angular_velocity_z',
            'imu_linear_acceleration_x', 'imu_linear_acceleration_y', 'imu_linear_acceleration_z',
            'imu_temperature',
            'gait_type',
            'position_mode',
            'velocity_mode',
            'yaw',
            'position_x', 'position_y', 'position_z',
            'yaw_speed',
            'velocity_x', 'velocity_y', 'velocity_z',
            'angular_speed',
            'foot_position_1', 'foot_position_2', 'foot_position_3', 'foot_position_4',
            'foot_contact_1', 'foot_contact_2', 'foot_contact_3', 'foot_contact_4',
            'foot_force_1', 'foot_force_2', 'foot_force_3', 'foot_force_4', 'foot_force_5', 'foot_force_6',
            'foot_force_7', 'foot_force_8', 'foot_force_9', 'foot_force_10', 'foot_force_11', 'foot_force_12',
            'foot_position_x1', 'foot_position_y1', 'foot_position_z1',
            'foot_position_x2', 'foot_position_y2', 'foot_position_z2',
            'foot_position_x3', 'foot_position_y3', 'foot_position_z3',
            'foot_position_x4', 'foot_position_y4', 'foot_position_z4',
            *[f'path_point_{i+1}_{field}' for i in range(10) for field in [
                'x', 'y', 'yaw', 'vx', 'vy', 'time'
            ]]
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
        if current_time - self.odom_start_time >= self.record_duration:
            self.open_new_odom_file()
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

            row = [
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
            self.odom_writer.writerow(row)
            if int(current_time * 1000) % 100 == 0:
                self.odom_csv.flush()
        except AttributeError as e:
            print(f"Error processing odom message: {e}")
        except IndexError as e:
            print(f"Index error in odom message: {e}")

    def low_callback(self, msg):
        current_time = time.time()
        if current_time - self.low_start_time >= self.record_duration:
            self.open_new_low_file()
        try:
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
            for motor in msg.motor_state[:35]:
                row += [
                    motor.mode, motor.q, motor.dq, motor.ddq, motor.tau_est,
                    motor.temperature[0], motor.temperature[1],
                    motor.sensor[0], motor.sensor[1],
                    motor.vol, motor.motorstate
                ] + list(motor.reserve)
            for _ in range(35 - len(msg.motor_state)):
                row += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
            row += list(msg.wireless_remote)
            row += list(msg.reserve)
            row += [msg.crc]
            self.low_writer.writerow(row)
            if int(current_time * 1000) % 100 == 0:
                self.low_csv.flush()
        except AttributeError as e:
            print(f"Error processing low message: {e}")
        except IndexError as e:
            print(f"Index error in low message: {e}")

    def run(self):
        ChannelFactoryInitialize(0)
        odom_sub = ChannelSubscriber("rt/odommodestate", SportModeState_, qos_depth=50)
        low_sub = ChannelSubscriber("rt/lowstate", LowState_, qos_depth=50)
        try:
            odom_sub.Init(self.odom_callback, 50)
            low_sub.Init(self.low_callback, 50)
            while True:
                time.sleep(1)
        except Exception as e:
            print(f"Error in subscriber: {e}")
        finally:
            self.__del__()

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