# -*- coding: utf-8 -*-
# 运行环境：Jetson Nano (边缘计算设备)
# 功能：双分类引擎调度 + 定向竹节检测 (仅前两路) + TensorRT 极限加速

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import socket
import struct
import cv2
import sys
import os

# ----------------- 核心配置区 -----------------
# ⚠️ 1. 正反面分类引擎 (用于相机 1, 2)
TOP_CLS_ENGINE_PATH = "/home/jetson1101/bamboo/UI/top_exp2.engine"

# ⚠️ 2. 侧面分类引擎 (用于相机 3, 4)
SIDE_CLS_ENGINE_PATH = "/home/jetson1101/bamboo/UI/side_exp3.engine"

# ⚠️ 3. 竹节目标检测引擎 (仅用于相机 1, 2)
DET_ENGINE_PATH = "/home/jetson1101/bamboo/UI/zhujie_det.engine"

CLASS_ID_OK = 1
DET_THRESHOLD = 0.5  # 竹节检测的置信度阈值

HOST = '127.0.0.1'
PORT = 8888
IMG_BYTES = 196608  # 256*256*3


class TRT_YOLO_Model:
    def __init__(self, engine_path, name="Model"):
        self.name = name
        print(f"[{self.name}] 正在加载 TensorRT 引擎...")
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 动态获取引擎的输入尺寸
        self.in_h, self.in_w = 256, 256
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                shape = self.engine.get_binding_shape(binding)
                self.in_h, self.in_w = shape[2], shape[3]
            else:
                self.out_shape = self.engine.get_binding_shape(binding)

        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})
        print(f"[{self.name}] 加载完毕！")

    def execute(self, img_bgr):
        img = cv2.resize(img_bgr, (self.in_w, self.in_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        np.copyto(self.inputs[0]["host"], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        return self.outputs[0]["host"]

    def predict_class(self, img_bgr):
        out = self.execute(img_bgr)
        class_id = int(np.argmax(out))
        return class_id, float(out[class_id])

    def predict_has_zhujie(self, img_bgr):
        try:
            out_flat = self.execute(img_bgr)
            out = out_flat.reshape(self.out_shape[1:])
            confs = np.max(out[4:, :], axis=0)
            return 1 if np.max(confs) > DET_THRESHOLD else 0
        except Exception as e:
            print(f"[{self.name}] 解析检测框失败: {e}")
            return 0


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def start_server():
    try:
        # 分别实例化 3 个 TRT 加速引擎
        classifier_top = TRT_YOLO_Model(TOP_CLS_ENGINE_PATH, "分类器(Top_C1/C2)")
        classifier_side = TRT_YOLO_Model(SIDE_CLS_ENGINE_PATH, "分类器(Side_C3/C4)")
        detector = TRT_YOLO_Model(DET_ENGINE_PATH, "检测器(ZhuJie)")
    except Exception as e:
        print(f"引擎加载失败: {e}")
        sys.exit(1)

    # 预热显卡
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    classifier_top.predict_class(dummy)
    classifier_side.predict_class(dummy)
    detector.predict_has_zhujie(dummy)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"\n🚀 TRT 二阶段级联服务(多工位协同版)已启动！端口 {PORT}")

        while True:
            conn, addr = s.accept()
            print(f"✅ UI 已连接: {addr}")

            with conn:
                while True:
                    for i in range(4):
                        img_data = recvall(conn, IMG_BYTES)
                        if not img_data: break
                        img = np.frombuffer(img_data, dtype=np.uint8).reshape((256, 256, 3))

                        # ================= 核心两阶段逻辑 =================
                        # 1. 根据工位切换分类模型
                        if i < 2:
                            cid, conf = classifier_top.predict_class(img)
                        else:
                            cid, conf = classifier_side.predict_class(img)
                        
                        has_zhujie = 0

                        # 2. 只有前两路 (i < 2) 且判断为 OK 时，才去检测竹节
                        if i < 2 and cid == CLASS_ID_OK:
                            has_zhujie = detector.predict_has_zhujie(img)

                        # 打包返回 (总共12字节)
                        conn.sendall(struct.pack('<ifi', cid, conf, has_zhujie))

                    if not img_data: break

if __name__ == "__main__":
    start_server()