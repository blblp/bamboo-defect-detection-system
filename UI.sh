#!/bin/bash

# ================= 配置区域 (修改为实际路径) =================
CONDA_INIT="/home/jetson1101/miniforge3/etc/profile.d/conda.sh"
# 定义conda环境名称/路径
CONDA_ENV="/home/jetson1101/miniforge3/envs/yolov8_py38"
# recv_data.py 所在的文件夹路径
RECV_DIR="/home/jetson1101/bamboo/communication"

SVM_DIR="/home/jetson1101/bamboo/UI"

SUDO_CMD_1='sudo -v'
SUDO_CMD_2='sudo chmod 666 /dev/ttyUSB0'
SUDO_CMD_3='sudo ifconfig eth0 down'
SUDO_CMD_4='sudo ifconfig eth0 192.168.0.36 netmask 255.255.255.0 up'
SUDO_PWD='123'
# ===================================================================

echo ">>> 开始初始化系统..."

#启动前，先清理所有残留的幽灵进程
echo ">>> 0. 正在清理历史残留的 TRT 幽灵进程..."
pkill -f trt_server.py
sleep 1

# 1. 获取 sudo 权限 
sudo -v
# expect -c "
# spawn $SUDO_CMD_1
# expect {
#     \"Password:\" { send \"$SUDO_PWD\r\"; exp_continue }
#     eof
# }
# "

# 2. 设置串口权限
echo ">>> 配置串口权限..."
# expect -c "
# spawn $SUDO_CMD_2
# expect {
#     \"Password:\" { send \"$SUDO_PWD\r\"; exp_continue }
#     eof
# }
# "
sudo chmod 666 /dev/ttyUSB0
# sudo chmod 666 /dev/ttyUSB1

# 3. 配置网络 IP
echo ">>> 配置网络 IP..."
# expect -c "
# spawn $SUDO_CMD_3
# expect {
#     \"Password:\" { send \"$SUDO_PWD\r\"; exp_continue }
#     eof
# }"

sudo ifconfig eth0 down

# expect -c "
# spawn $SUDO_CMD_4
# expect {
#     \"Password:\" { send \"$SUDO_PWD\r\"; exp_continue }
#     eof
# }"

sudo ifconfig eth0 192.168.0.36 netmask 255.255.255.0 up

# 4. 配置显示
echo ">>> 配置 Display..."
export DISPLAY=:0
xhost +local:root

# ========================== 启动程序 ==========================
echo ">>> 1. 正在唤醒底层 C++ TensorRT 推理服务 (原生 Python3)..."
# 重新引入一下环境变量，防止 sudo 弄丢路径
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 弹出一个新的独立终端来运行 trt_server，方便实时监控方差和检测日志
gnome-terminal -- bash -c "python3 /home/jetson1101/bamboo/UI/trt_server.py; exec bash"

echo ">>> 等待 TensorRT 引擎预热 ..."
sleep 5

echo ">>> 2. 正在启动前端监控 UI 界面 ..."
source ~/.bashrc
source /home/jetson1101/miniforge3/etc/profile.d/conda.sh
cd "$(dirname "$0")"
conda activate yolov8_py38
python /home/jetson1101/bamboo/UI/exe-2.py

# ========================== 退出清理 ==========================
echo ">>> UI 界面已关闭，正在同步清理底层驻留的 TRT 服务..."

pkill -f trt_server.py
echo ">>> 所有程序已安全退出，系统已释放显存！"
