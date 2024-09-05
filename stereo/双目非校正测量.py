import cv2
import numpy as np
import glob
def tstR(imagesPath):
    chessboard_size = (11, 8)  # 棋盘格的内角点个数
    square_size = 3.0  # 棋盘格每个方格的实际大小，单位可以是毫米、厘米或米

    # 准备棋盘格的世界坐标系坐标（假设z=0）
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 用于存储所有图像的世界坐标系点和图像坐标系点
    objpoints = []  # 世界坐标系中的点
    imgpoints = []  # 图像坐标系中的点

    # 获取所有棋盘格图像的路径
    images = imagesPath

    for image_path in images:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # 执行相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 将旋转向量转换为旋转矩阵
    rvec_matrix = cv2.Rodrigues(rvecs[0])[0]  # 只取第一个图像的旋转矩阵

    # 返回相机内参、畸变系数和旋转矩阵
    return mtx, dist, rvec_matrix

# 读取左右图像
left_image = cv2.imread('left.jpg', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right.jpg', cv2.IMREAD_GRAYSCALE)

# 获取左右相机图片路径
left_Path = glob.glob('left/*.bmp')
right_Path = glob.glob('right/*.bmp')

# 相机内参
K1, D1, R1 = tstR(left_Path)
K2, D2, R2 = tstR(right_Path)
# 计算旋转矩阵和平移向量
R1_i = np.linalg.inv(R1)  # 现在R1是3x3的旋转矩阵，可以求逆
R = R2.dot(R1_i)
baseline = 12.0
T = np.array([baseline]).reshape(-1, 1)
# 读取左右相机的图像
img_left = cv2.imread('left/1.bmp')
img_right = cv2.imread('right/1.bmp')
# 转为灰度图
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# 棋盘格的尺寸 (内角点数量)
pattern_size = (12, 9)

# 查找棋盘格的角点
ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

if ret_left and ret_right:
    # 优化角点位置（亚像素精度）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

    # 去畸变
    corners_left_undist = cv2.undistortPoints(corners_left, K1, D1)
    corners_right_undist = cv2.undistortPoints(corners_right, K2, D2)

    # 构造投影矩阵
    P1 = np.hstack((K1, np.zeros((3, 1))))  # 左相机投影矩阵 [K1|0]
    P2 = np.hstack((K2, np.dot(K2, np.hstack((R, T.reshape(-1, 1))))))  # 右相机投影矩阵 [K2|R|T]

    # 使用角点进行三角测量
    points_4D = cv2.triangulatePoints(P1, P2, corners_left_undist.T, corners_right_undist.T)

    # 将齐次坐标转换为 3D 点
    points_3D = points_4D[:3, :] / points_4D[3, :]

    # 输出 3D 点
    print("3D Points: \n", points_3D.T)
else:
    print("未能检测到棋盘格角点")
