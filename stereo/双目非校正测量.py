import cv2
import numpy as np

# 相机内参矩阵和畸变系数
intrinsic_matrix_left = np.array([
    [1393.22172200895, -0.995545249279859, 955.251005896264],
    [0, 1393.64853177193, 533.727854721113],
    [0, 0, 1]
])
intrinsic_matrix_right = np.array([
    [1392.85086761789, -1.11152594052554, 941.378478650626],
    [0, 1392.25346501003, 530.138196689747],
    [0, 0, 1]
])

# 基线长度（单位：毫米）
baseline = 120  # 单位: 毫米

# 焦距
f_x_left = intrinsic_matrix_left[0, 0]
f_y_left = intrinsic_matrix_left[1, 1]

f_x_right = intrinsic_matrix_right[0, 0]
f_y_right = intrinsic_matrix_right[1, 1]

# 读取图像
img_left = cv2.imread('left_1.png')
img_right = cv2.imread('right_1.png')

# 棋盘格参数
pattern_size = (12,9)  # 棋盘格内角点的数量（列数，行数）

# 棋盘格角点检测
ret_left, corners_left = cv2.findChessboardCorners(img_left, pattern_size)
ret_right, corners_right = cv2.findChessboardCorners(img_right, pattern_size)

if ret_left and ret_right:
    # 角点坐标转换为浮点数
    corners_left = np.float32(corners_left)
    corners_right = np.float32(corners_right)

    # 计算视差
    disparity = np.abs(corners_left[:, :, 0] - corners_right[:, :, 0])

    # 计算深度
    Z = (f_x_left * baseline) / disparity

    # 计算三维坐标
    X = (corners_left[:, :, 0] - intrinsic_matrix_left[0, 2]) * Z / f_x_left
    Y = (corners_left[:, :, 1] - intrinsic_matrix_left[1, 2]) * Z / f_y_left

    # 显示结果
    for i in range(len(X)):
        print(f'Corner {i}: X = {X[i][0]}, Y = {Y[i][0]}, Z = {Z[i][0]}')
else:
    print("无法检测到棋盘格角点")
