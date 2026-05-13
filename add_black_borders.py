import os
import cv2
import numpy as np

# ======================== 配置区 ========================
# 原始数据集路径
INPUT_DIR = "/media/jfyang/软件/wxn/bamboo_dataset_3class2_noblack"

# 自动生成的新数据集路径
OUTPUT_DIR = "/media/jfyang/软件/wxn/bamboo_dataset_3class2"

# YOLO 模型期待的正方形尺寸
TARGET_SIZE = 256

# 支持的图片后缀
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
# ========================================================

def letterbox_image(img_bgr, target_size):
    """等比例缩放图片，并在上下填充纯黑边使其成为正方形"""
    h, w = img_bgr.shape[:2]
    
    # 1. 计算缩放比例，保证最长边刚好放进 target_size
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 2. 等比例缩放图片 
    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 3. 创建一个纯黑的正方形画布
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 4. 计算居中贴图的左上角坐标
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    
    # 5. 把缩放后的竹筷子贴到黑画布正中间
    canvas[top:top+new_h, left:left+new_w] = img_resized
    
    return canvas

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"找不到输入文件夹: {INPUT_DIR}")
        return

    print(f"开始为数据集添加黑边 (Letterbox)...")
    print(f"数据源: {INPUT_DIR}")
    print(f"输出至: {OUTPUT_DIR}\n")

    processed_count = 0

    # 递归遍历整个数据集文件夹
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                # 原始图片的完整路径
                img_path = os.path.join(root, file)
                
                # 计算相对路径
                rel_path = os.path.relpath(root, INPUT_DIR)
                out_folder = os.path.join(OUTPUT_DIR, rel_path)
                
                # 建立输出子文件夹
                os.makedirs(out_folder, exist_ok=True)
                
                # 输出图片的完整路径
                out_path = os.path.join(out_folder, file)

                # 读取、处理、保存
                img = cv2.imread(img_path)
                if img is not None:
                    img_padded = letterbox_image(img, TARGET_SIZE)
                    cv2.imwrite(out_path, img_padded)
                    processed_count += 1
                    
                    # 每处理 500 张打印一次进度
                    if processed_count % 500 == 0:
                        print(f"⏳ 已处理 {processed_count} 张图片...")

    print("-" * 50)
    print(f"共计为 {processed_count} 张图片添加了工业级黑边。")
    print(f"使用新文件夹进行训练: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
