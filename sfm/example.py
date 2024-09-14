import cv2

# 初始化视频捕获对象
video_path = 'D:/Desktop/data/WIN_20221205_15_04_01_Pro.mp4'  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error opening video file")
    exit()

# 设置保存图片的路径和格式
output_dir = 'output_images/'  # 指定输出图片的目录
frame_count = 0  # 初始化帧计数器
save_count = 0  # 初始化保存计数器

# 创建输出目录（如果不存在）
import os
os.makedirs(output_dir, exist_ok=True)

# 读取并保存每一帧
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果无法读取到帧，则退出循环

    frame_count += 1

    # 每五帧保存一次
    if frame_count % 5 == 0:
        # 保存当前帧
        frame_filename = f"{output_dir}frame_{save_count:04d}.jpg"  # 格式化文件名
        cv2.imwrite(frame_filename, frame)
        save_count += 1

# 释放资源
cap.release()
print(f"Total frames saved: {save_count}")
