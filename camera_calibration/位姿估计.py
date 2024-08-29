import cv2
import numpy as np

# 读取相机标定参数
with np.load('data.npz') as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

# 定义棋盘格的大小和角点数
chessboard_size = (6, 13)
square_size = 18  # 每个方格的边长

# 准备棋盘格的3D点对象 (假设棋盘格在z=0平面上)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 读取图像
img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 寻找棋盘格角点
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

if ret:
    # 计算相机位姿
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)

    # 将旋转向量转化为旋转矩阵
    rmat, _ = cv2.Rodrigues(rvecs)

    # 打印相机的位姿
    print("旋转矩阵:")
    print(rmat)
    print("\n位移向量:")
    print(tvecs)

    # 计算相机姿态
    # 相机在世界坐标系中的位置
    camera_position = -np.matrix(rmat).T * np.matrix(tvecs)
    print("\n相机位置 (世界坐标系):")
    print(camera_position)

    # 将旋转矩阵转为欧拉角（绕x, y, z轴的旋转角度）
    theta_x = np.arctan2(rmat[2, 1], rmat[2, 2])
    theta_y = np.arcsin(-rmat[2, 0])
    theta_z = np.arctan2(rmat[1, 0], rmat[0, 0])

    print("\n相机朝向 (欧拉角，单位：弧度):")
    print("theta_x: ", theta_x)
    print("theta_y: ", theta_y)
    print("theta_z: ", theta_z)

else:
    print("未找到棋盘格角点")
