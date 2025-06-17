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




pip install numpy
pip install cvxpy
pip install pyyaml
pip install trimesh
pip install pin
pip install urdf_parser_py
pip install Mosek
```
You also need to add a [license](https://www.mosek.com/products/academic-licenses/) for MOSEK in order to use the solver.  

git add .
git commit -m ""
git push origin main

