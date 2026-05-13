# -*- coding: utf-8 -*-
import sys
import os
import time
import numpy as np
import cv2
import serial
import socket
import struct
from collections import deque
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

# ======================== 核心配置 ========================
MV_IMPORT_PATH = "/home/jetson1101/bamboo/UI/MvImport"
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

CONF_THRESHOLD = 0.8
CLASS_ID_OK = 1

# 流水线物理距离延迟补偿 (队列长度)
CAMERA_DISTANCE_STEPS = 6
SIDE_CAMERA_STEPS = 2

VISUAL_WIDTH = 630
VISUAL_HEIGHT = 440

sys.path.insert(0, MV_IMPORT_PATH)
try:
    from MvCameraControl_class import *
except ImportError as e:
    QMessageBox.critical(None, "导入错误", f"无法导入相机类：{e}")
    sys.exit(1)


def get_letterbox_square(img_bgr, target_size=256):
    h, w = img_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img_resized
    return canvas


def get_res_string(cid, conf, zj):
    if conf < CONF_THRESHOLD: return "Uncertain"
    if cid != CLASS_ID_OK: return "NG"
    if zj == 1: return "OK(ZhuJie)"
    return "OK(Pure)"


class MVS_Camera:
    def __init__(self, device_info):
        self.device_info = device_info
        self.cam = MvCamera()
        self.handle_created = False
        self.is_open = False
        self.data_buf = None
        self.buf_size = 0

        ret = self.cam.MV_CC_CreateHandle(device_info)
        if ret != 0: raise Exception(f"创建句柄失败：0x{ret:x}")
        self.handle_created = True

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0: raise Exception(f"打开相机失败：0x{ret:x}")
        self.is_open = True

        if device_info.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if nPacketSize > 0: self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0: raise Exception(f"启动抓流失败：0x{ret:x}")

    def set_trigger_mode(self, on=True):
        if not self.is_open: return
        if on:
            self.cam.MV_CC_SetEnumValue("TriggerMode", 1)
            self.cam.MV_CC_SetEnumValue("TriggerSource", 7)
        else:
            self.cam.MV_CC_SetEnumValue("TriggerMode", 0)

    def software_trigger(self):
        if not self.is_open: return
        self.cam.MV_CC_SetCommandValue("TriggerSoftware")

    def get_frame(self, timeout=1000):
        try:
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

            self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
            nPayloadSize = stParam.nCurValue

            if self.data_buf is None or nPayloadSize > self.buf_size:
                self.data_buf = (c_ubyte * nPayloadSize)()
                self.buf_size = nPayloadSize

            ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.data_buf), nPayloadSize, stFrameInfo, timeout)
            if ret != 0: return None

            h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
            nRGBSize = w * h * 3
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            img_buff = (c_ubyte * nRGBSize)()
            stConvertParam.nWidth = w
            stConvertParam.nHeight = h
            stConvertParam.pSrcData = self.data_buf
            stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
            stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
            stConvertParam.pDstBuffer = img_buff
            stConvertParam.nDstBufferSize = nRGBSize
            self.cam.MV_CC_ConvertPixelType(stConvertParam)

            img_np = np.frombuffer(img_buff, dtype=np.uint8).reshape((h, w, 3))
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def close(self):
        if self.is_open:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.is_open = False
        if self.handle_created:
            self.cam.MV_CC_DestroyHandle()
            self.handle_created = False


class SerialWorker(QThread):
    result_signal = pyqtSignal(object, object, object, object, str, str, str, str, str, float, float, float, float)
    log_signal = pyqtSignal(str)

    def __init__(self, main_ui):
        super().__init__()
        self.main_ui = main_ui
        self.is_running = False
        self.client_sock = None

    def run(self):
        self.is_running = True
        ser = self.main_ui.ser

        if self.main_ui.is_detecting:
            try:
                self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_sock.connect(('127.0.0.1', 8888))
                self.log_signal.emit(f"成功连接到边缘推理核心！")
            except Exception as e:
                self.log_signal.emit(f"连接失败 ({e})")
                self.is_running = False
                return

        while self.is_running:
            if ser and ser.is_open:
                try:
                    incoming = ser.read(1)
                    if incoming == b'U':
                        if self.main_ui.camera1: self.main_ui.camera1.software_trigger()
                        if self.main_ui.camera2: self.main_ui.camera2.software_trigger()
                        if self.main_ui.camera3: self.main_ui.camera3.software_trigger()
                        if self.main_ui.camera4: self.main_ui.camera4.software_trigger()

                        frame1 = self.main_ui.camera1.get_frame(timeout=500) if self.main_ui.camera1 else None
                        frame2 = self.main_ui.camera2.get_frame(timeout=500) if self.main_ui.camera2 else None
                        frame3 = self.main_ui.camera3.get_frame(timeout=500) if self.main_ui.camera3 else None
                        frame4 = self.main_ui.camera4.get_frame(timeout=500) if self.main_ui.camera4 else None

                        if frame1 is None or frame2 is None or frame3 is None or frame4 is None:
                            continue

                        if self.main_ui.is_detecting and self.client_sock:
                            img1_ready = np.ascontiguousarray(get_letterbox_square(frame1, 256))
                            img2_ready = np.ascontiguousarray(get_letterbox_square(frame2, 256))
                            img3_ready = np.ascontiguousarray(get_letterbox_square(frame3, 256))
                            img4_ready = np.ascontiguousarray(get_letterbox_square(frame4, 256))

                            self.client_sock.sendall(img1_ready.tobytes())
                            r1 = self.client_sock.recv(12)
                            self.client_sock.sendall(img2_ready.tobytes())
                            r2 = self.client_sock.recv(12)
                            self.client_sock.sendall(img3_ready.tobytes())
                            r3 = self.client_sock.recv(12)
                            self.client_sock.sendall(img4_ready.tobytes())
                            r4 = self.client_sock.recv(12)

                            if len(r1) == 12 and len(r2) == 12 and len(r3) == 12 and len(r4) == 12:
                                id1, conf1, zj1 = struct.unpack('<ifi', r1)
                                id2, conf2, zj2 = struct.unpack('<ifi', r2)
                                id3, conf3, zj3 = struct.unpack('<ifi', r3)
                                id4, conf4, zj4 = struct.unpack('<ifi', r4)

                                curr_is_ok1 = (id1 == CLASS_ID_OK) if conf1 >= CONF_THRESHOLD else False

                                # 处理物理流水线产生的时序延迟对齐
                                past_id2, past_conf2, past_zj2 = self.main_ui.cam2_queue.popleft()
                                past_id3, past_conf3, past_zj3 = self.main_ui.cam3_queue.popleft()
                                past_id4, past_conf4, past_zj4 = self.main_ui.cam4_queue.popleft()

                                past_is_ok2 = (past_id2 == CLASS_ID_OK) if past_conf2 >= CONF_THRESHOLD else False
                                past_is_ok3 = (past_id3 == CLASS_ID_OK) if past_conf3 >= CONF_THRESHOLD else False
                                past_is_ok4 = (past_id4 == CLASS_ID_OK) if past_conf4 >= CONF_THRESHOLD else False

                                self.main_ui.cam2_queue.append((id2, conf2, zj2))
                                self.main_ui.cam3_queue.append((id3, conf3, zj3))
                                self.main_ui.cam4_queue.append((id4, conf4, zj4))

                                final_is_ok = curr_is_ok1 and past_is_ok2 and past_is_ok3 and past_is_ok4

                                if conf1 < CONF_THRESHOLD:
                                    send_data, grade = b'D', 'D'
                                elif final_is_ok:
                                    # 此时 zj3 和 zj4 必为 0 (因为后端逻辑拦截了侧面的竹节检测)
                                    has_zhujie = (zj1 == 1) or (past_zj2 == 1) or (past_zj3 == 1) or (past_zj4 == 1)
                                    if has_zhujie:
                                        send_data, grade = b'B', 'B'
                                    else:
                                        send_data, grade = b'A', 'A'
                                else:
                                    send_data, grade = b'C', 'C'

                                ser.write(send_data)

                                self.result_signal.emit(
                                    frame1, frame2, frame3, frame4, 
                                    get_res_string(id1, conf1, zj1), get_res_string(id2, conf2, zj2), 
                                    get_res_string(id3, conf3, zj3), get_res_string(id4, conf4, zj4), 
                                    grade, conf1, conf2, conf3, conf4
                                )
                except Exception as e:
                    self.log_signal.emit(f"线程错误: {e}")
            else:
                time.sleep(0.01)

    def stop(self):
        self.is_running = False
        if self.client_sock:
            self.client_sock.close()
            self.client_sock = None


class Main_ui(QMainWindow):
    def __init__(self):
        super(Main_ui, self).__init__()
        self.log_signal_handler = lambda msg: print(f"【系统日志】{msg}")

        self.cam2_queue = deque([(CLASS_ID_OK, 1.0, 0)] * CAMERA_DISTANCE_STEPS, maxlen=CAMERA_DISTANCE_STEPS)
        self.cam3_queue = deque([(CLASS_ID_OK, 1.0, 0)] * SIDE_CAMERA_STEPS, maxlen=SIDE_CAMERA_STEPS)
        self.cam4_queue = deque([(CLASS_ID_OK, 1.0, 0)] * SIDE_CAMERA_STEPS, maxlen=SIDE_CAMERA_STEPS)

        self.setupUi(self)
        self.camera1 = self.camera2 = self.camera3 = self.camera4 = None

        self.count_total = self.count_A = self.count_B = self.count_C = 0

        self.preview_timer = QTimer()
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self.update_preview)

        self.serial_worker = SerialWorker(self)
        self.serial_worker.result_signal.connect(self.handle_worker_result)
        self.serial_worker.log_signal.connect(self.log_signal_handler)

        self.is_camera_open = self.is_detecting = False

        self.OpenCamera.clicked.connect(self.on_open_camera)
        self.BeginDetect.clicked.connect(self.on_begin_detect)
        self.StopDetect.clicked.connect(self.on_stop_detect)
        self.ShutDown.clicked.connect(self.on_shut_down)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_time)
        self.timer.start(1000)
        self.update_system_time()

        self.ser = None
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            self.log_signal_handler(f"串口初始化成功")
        except Exception as e:
            QMessageBox.warning(self, "串口警告", f"串口打开失败：{e}")

    def get_color_from_str(self, res_str):
        if "ZhuJie" in res_str: return (0, 165, 255)
        if "OK" in res_str: return (0, 255, 0)
        return (0, 0, 255)

    def draw_frame(self, frame, label, color):
        h, w = frame.shape[:2]
        scale = min(VISUAL_WIDTH / w, VISUAL_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        display_frame = np.zeros((VISUAL_HEIGHT, VISUAL_WIDTH, 3), dtype=np.uint8)
        top = (VISUAL_HEIGHT - new_h) // 2
        left = (VISUAL_WIDTH - new_w) // 2
        display_frame[top:top + new_h, left:left + new_w] = img_resized
        cv2.rectangle(display_frame, (0, 0), (VISUAL_WIDTH - 1, VISUAL_HEIGHT - 1), color, 6)
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_frame.data, VISUAL_WIDTH, VISUAL_HEIGHT, VISUAL_WIDTH * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_image))

    def update_work_mode(self):
        if self.is_detecting and not self.serial_worker.is_running:
            self.preview_timer.stop()
            if self.camera1: self.camera1.set_trigger_mode(True)
            if self.camera2: self.camera2.set_trigger_mode(True)
            if self.camera3: self.camera3.set_trigger_mode(True)
            if self.camera4: self.camera4.set_trigger_mode(True)
            self.serial_worker.start()
        elif not self.is_detecting and self.serial_worker.is_running:
            self.serial_worker.stop()
            self.serial_worker.wait()
            if self.camera1: self.camera1.set_trigger_mode(False)
            if self.camera2: self.camera2.set_trigger_mode(False)
            if self.camera3: self.camera3.set_trigger_mode(False)
            if self.camera4: self.camera4.set_trigger_mode(False)
            self.preview_timer.start()

    def update_preview(self):
        if not self.is_camera_open: return
        for cam, label in zip([self.camera1, self.camera2, self.camera3, self.camera4],
                              [self.image_1, self.image_2, self.image_3, self.image_4]):
            if cam:
                f = cam.get_frame(timeout=100)
                if f is not None: self.draw_frame(f, label, (255, 191, 0))

    def handle_worker_result(self, f1, f2, f3, f4, s1, s2, s3, s4, grade, c1, c2, c3, c4):
        if grade != 'D':
            self.count_total += 1
            if grade == 'A':
                self.count_A += 1
                self.radio_btn_A.setChecked(True)
            elif grade == 'B':
                self.count_B += 1
                self.radio_btn_B.setChecked(True)
            elif grade == 'C':
                self.count_C += 1
                self.radio_btn_C.setChecked(True)

            self.lineEdit_count.setText(str(self.count_total))
            self.label_A.setText(f"{self.count_A}个")
            self.label_B.setText(f"{self.count_B}个")
            self.label_C.setText(f"{self.count_C}个")

        if f1 is not None: self.draw_frame(f1, self.image_1, self.get_color_from_str(s1))
        if f2 is not None: self.draw_frame(f2, self.image_2, self.get_color_from_str(s2))
        if f3 is not None: self.draw_frame(f3, self.image_3, self.get_color_from_str(s3))
        if f4 is not None: self.draw_frame(f4, self.image_4, self.get_color_from_str(s4))

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1550, 1020)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        self.widget_left = QtWidgets.QWidget(self.centralwidget)
        self.widget_left.setGeometry(QtCore.QRect(10, 10, 200, 950))
        self.widget_left.setStyleSheet("background-color: #f0f0f0;")

        self.OpenCamera = QtWidgets.QPushButton("打开摄像机", self.widget_left)
        self.OpenCamera.setGeometry(QtCore.QRect(20, 20, 160, 40))

        self.BeginDetect = QtWidgets.QPushButton("开始检测", self.widget_left)
        self.BeginDetect.setGeometry(QtCore.QRect(20, 70, 160, 40))
        self.BeginDetect.setEnabled(False)

        self.StopDetect = QtWidgets.QPushButton("停止检测", self.widget_left)
        self.StopDetect.setGeometry(QtCore.QRect(20, 120, 160, 40))
        self.StopDetect.setEnabled(False)

        self.ShutDown = QtWidgets.QPushButton("关闭系统", self.widget_left)
        self.ShutDown.setGeometry(QtCore.QRect(20, 470, 160, 40))

        self.label_count = QtWidgets.QLabel("检测总数: ", self.widget_left)
        self.label_count.setGeometry(QtCore.QRect(20, 530, 80, 30))
        self.lineEdit_count = QtWidgets.QLineEdit("0", self.widget_left)
        self.lineEdit_count.setGeometry(QtCore.QRect(100, 530, 80, 30))
        self.lineEdit_count.setReadOnly(True)

        self.label_grade = QtWidgets.QLabel("品类级别：", self.widget_left)
        self.label_grade.setGeometry(QtCore.QRect(20, 580, 80, 30))

        self.radio_btn_A = QtWidgets.QRadioButton("优等品(A)", self.widget_left)
        self.radio_btn_A.setGeometry(QtCore.QRect(20, 620, 100, 30))
        self.label_A = QtWidgets.QLabel("0个", self.widget_left)
        self.label_A.setGeometry(QtCore.QRect(120, 620, 60, 30))

        self.radio_btn_B = QtWidgets.QRadioButton("二等品(B)", self.widget_left)
        self.radio_btn_B.setGeometry(QtCore.QRect(20, 660, 100, 30))
        self.label_B = QtWidgets.QLabel("0个", self.widget_left)
        self.label_B.setGeometry(QtCore.QRect(120, 660, 60, 30))

        self.radio_btn_C = QtWidgets.QRadioButton("废品(NG)", self.widget_left)
        self.radio_btn_C.setGeometry(QtCore.QRect(20, 700, 100, 30))
        self.label_C = QtWidgets.QLabel("0个", self.widget_left)
        self.label_C.setGeometry(QtCore.QRect(120, 700, 60, 30))
        self.radio_btn_A.setChecked(True)

        self.widget_right = QtWidgets.QWidget(self.centralwidget)
        self.widget_right.setGeometry(QtCore.QRect(220, 10, 1320, 980))
        self.widget_right.setStyleSheet("background-color: #e0e0e0;")

        self.label_time = QtWidgets.QLabel("系统时间：", self.widget_right)
        self.label_time.setGeometry(QtCore.QRect(1100, 10, 200, 30))

        self.image_1 = QtWidgets.QLabel("C1 判定工位...", self.widget_right)
        self.image_1.setGeometry(QtCore.QRect(20, 50, VISUAL_WIDTH, VISUAL_HEIGHT))
        self.image_1.setStyleSheet("background-color: black;")

        self.image_2 = QtWidgets.QLabel("C2 上游记录...", self.widget_right)
        self.image_2.setGeometry(QtCore.QRect(670, 50, VISUAL_WIDTH, VISUAL_HEIGHT))
        self.image_2.setStyleSheet("background-color: black;")

        self.image_3 = QtWidgets.QLabel("C3 左侧...", self.widget_right)
        self.image_3.setGeometry(QtCore.QRect(20, 510, VISUAL_WIDTH, VISUAL_HEIGHT))
        self.image_3.setStyleSheet("background-color: black;")

        self.image_4 = QtWidgets.QLabel("C4 右侧...", self.widget_right)
        self.image_4.setGeometry(QtCore.QRect(670, 510, VISUAL_WIDTH, VISUAL_HEIGHT))
        self.image_4.setStyleSheet("background-color: black;")

        MainWindow.setCentralWidget(self.centralwidget)

    def update_system_time(self):
        self.label_time.setText(f"系统时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

    def on_open_camera(self):
        if self.is_camera_open: return
        try:
            deviceList = MV_CC_DEVICE_INFO_LIST()
            ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
            if ret != 0: raise Exception(f"枚举相机失败：0x{ret:x}")

            CAM2_SN = "DA6567823"
            CAM1_SN = "DA9064874"
            CAM3_SN = "DA9861304"
            CAM4_SN = "DA9861292"

            for i in range(deviceList.nDeviceNum):
                device_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                sn_bytes = device_info.SpecialInfo.stGigEInfo.chSerialNumber if device_info.nTLayerType == MV_GIGE_DEVICE else device_info.SpecialInfo.stUsb3VInfo.chSerialNumber
                sn_str = "".join([chr(b) for b in sn_bytes if b != 0])

                if sn_str == CAM1_SN: self.camera1 = MVS_Camera(device_info)
                elif sn_str == CAM2_SN: self.camera2 = MVS_Camera(device_info)
                elif sn_str == CAM3_SN: self.camera3 = MVS_Camera(device_info)
                elif sn_str == CAM4_SN: self.camera4 = MVS_Camera(device_info)

            self.is_camera_open = True
            self.BeginDetect.setEnabled(True)
            self.preview_timer.start()
            self.log_signal_handler("相机已打开，进入预览模式")
        except Exception as e:
            QMessageBox.critical(self, "失败", f"相机打开失败：{e}")

    def on_begin_detect(self):
        if self.is_detecting: return
        self.cam2_queue.clear()
        self.cam3_queue.clear()
        self.cam4_queue.clear()
        self.cam2_queue.extend([(CLASS_ID_OK, 1.0, 0)] * CAMERA_DISTANCE_STEPS)
        self.cam3_queue.extend([(CLASS_ID_OK, 1.0, 0)] * SIDE_CAMERA_STEPS)
        self.cam4_queue.extend([(CLASS_ID_OK, 1.0, 0)] * SIDE_CAMERA_STEPS)

        self.count_total = self.count_A = self.count_B = self.count_C = 0
        self.lineEdit_count.setText("0")
        self.label_A.setText("0个")
        self.label_B.setText("0个")
        self.label_C.setText("0个")

        self.is_detecting = True
        self.BeginDetect.setEnabled(False)
        self.StopDetect.setEnabled(True)
        self.update_work_mode()

    def on_stop_detect(self):
        self.is_detecting = False
        self.BeginDetect.setEnabled(True)
        self.StopDetect.setEnabled(False)
        self.update_work_mode()

    def on_shut_down(self):
        if QMessageBox.question(self, "确认", "是否确定关闭？", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.preview_timer.stop()
            if self.serial_worker.is_running:
                self.serial_worker.stop()
                self.serial_worker.wait()
            QApplication.quit()

    def closeEvent(self, event):
        self.on_shut_down()
        event.accept()

if __name__ == "__main__":
    if os.environ.get("DISPLAY") is None: os.environ["DISPLAY"] = ":0"
    app = QApplication(sys.argv)
    ui = Main_ui()
    ui.setWindowTitle("竹片极速四摄系统 ")
    ui.show()
    sys.exit(app.exec_())
