import cv2
import numpy as np
import glob
import os


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


def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output

# 创建result文件夹（如果不存在）
if not os.path.exists('result'):
    os.makedirs('result')

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
T = np.array([-12, 0.0, 0.0], dtype=np.float64)  # 平移向量

print("旋转矩阵 R:\n", R)
print("平移向量 T:\n", T)

# 焦距
f_x = K1[0, 0]

# 图像尺寸
h1, w1 = left_image.shape[:2]
h2, w2 = right_image.shape[:2]

# 立体校正
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, (w1, h1), R, T, alpha=0)

# 初始化重映射
left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, None, R1, P1, (w1, h1), cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, None, R2, P2, (w2, h2), cv2.CV_16SC2)

# 立体校正后的图像
rectified_left = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
rectified_right = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)

# 绘制校正对比线
line = draw_line(left_image, rectified_left)
cv2.imwrite('result/AAA.png', line)

cv2.imwrite('result/left_rectified.png', rectified_left)
cv2.imwrite('result/right_rectified.png', rectified_right)

# 计算视差图
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5)
disparity = stereo.compute(rectified_left, rectified_right)

# 视差图归一化处理（可视化）
disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_norm = np.uint8(disparity_norm)

# 保存视差图
cv2.imwrite('result/disparity.png', disparity_norm)

# 计算深度图
depth = (f_x * baseline) / (disparity + 1e-6)

# 深度图归一化处理（可视化）
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = np.uint8(depth_norm)

# 保存深度图
cv2.imwrite('result/depth.png', depth_norm)
