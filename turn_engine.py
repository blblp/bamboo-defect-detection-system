from ultralytics import YOLO


def export_to_tensorrt():
    print(" 开始转换模型为 TensorRT engine 格式...")

    # 1. 加载训练好的 pt 模型
    model_path = "/media/jfyang/软件/wxn/bamboo/bamboo_models/2class/exp2/weights/best.pt"
    model = YOLO(model_path)

    # 2. 执行导出操作
    model.export(
        format='engine',  # 核心参数：指定导出为 TensorRT 格式
        imgsz=256,  #  与训练时设置的 imgsz 保持一致
        half=False,  # 开启 FP16 半精度压缩，成倍提升推理速度且几乎不掉精度
        device=0,  # 指定使用 GPU 进行转换
        workspace=4,  # 分配给 TensorRT 转换过程的最大显存 (GB)，默认4G一般够用
        simplify=True  # 简化模型结构，有助于提升最终引擎的运行效率
    )

    print(f"\n 转换完成！\n 生成的 .engine 文件已自动保存在原 pt 文件的同级目录下。")


if __name__ == '__main__':
    export_to_tensorrt()
