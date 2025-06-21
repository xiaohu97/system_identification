# system_identification
A Python package for physically consistent inertial parameters identification of legged robots using joint torque measurements.

### Dependencies
numpy, cvxpy, yaml, trimesh, pinocchio, urdf_parser_py, MOSEK solver.
```
conda create -n ustc_identification python=3.10
conda activate ustc_identification 
conda install numpy pyyaml trimesh pinocchio cvxpy
# 安装仅限 Pip 的包
pip install urdf_parser_py
pip install Mosek  # 仅在拥有 MOSEK 许可证时

### git 
```
git add .
git commit -m ""
git push origin main
```

### 收集数据
记录宇树G1数据 、 时间、位置、速度、加速度、力矩、触地状态 
500hz 20000个数据  = 40 s
#  1.使用ros2接口记录
```
ros2 topic list
ros2 bag record -a -o spot_data -d 40 
```

#  2.使用dds接口记录
```
python deploy_real.py enp4s0 g1.yaml

python3 subscriber.py

python3 read_g1_data_logger.py output_g1
# python3 read_g1_data_logger.py odom_data low_data
```

#  需要的数据
    # timestamp [s] and [10^{-9}s]
    # - seconds
    # - nanoseconds    

    # position [m]
    # - base:   body_lin_x	body_lin_y	body_lin_z	body_ang_x	body_ang_y	body_ang_z	body_ang_w
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
    # https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_state#bosdyn.client.robot_state.RobotStateStreamingClient.get_robot_state_stream
    # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#robotstatestreamingservice
    