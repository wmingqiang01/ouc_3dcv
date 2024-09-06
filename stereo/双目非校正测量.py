import cv2
import numpy as np
import glob


# 相机标定函数
def tstR(imagesPath):
    chessboard_size = (11, 8)  # 棋盘格内角点数
    square_size = 3.0  # 棋盘格方块实际大小

    # 定义棋盘格的世界坐标系点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = imagesPath

    for image_path in images:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    rvec_matrix = cv2.Rodrigues(rvecs[0])[0]

    return mtx, dist, rvec_matrix


# 获取图片路径
left_Path = glob.glob('left/*.bmp')
right_Path = glob.glob('right/*.bmp')

# 标定左右相机
K1, D1, R1 = tstR(left_Path)
K2, D2, R2 = tstR(right_Path)

# 计算旋转和平移矩阵
R1_i = np.linalg.inv(R1)
R = R2.dot(R1_i)
T = np.array([-12, 0.0, 0.0], dtype=np.float64).reshape(3, 1)

# 读取左右相机的图像
img_left = cv2.imread('left/1.bmp')
img_right = cv2.imread('right/1.bmp')
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# 棋盘格尺寸
pattern_size = (11, 8)

# 查找角点
ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

if ret_left and ret_right:
    # 亚像素优化角点
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

    # 手动绘制角点
    img_left_with_corners = img_left.copy()
    img_right_with_corners = img_right.copy()

    for corner in corners_left:
        cv2.circle(img_left_with_corners, tuple(corner.ravel().astype(int)), 10, (0, 255, 0), -1)

    for corner in corners_right:
        cv2.circle(img_right_with_corners, tuple(corner.ravel().astype(int)), 10, (0, 255, 0), -1)

    # 保存带有角点标注的图像
    cv2.imwrite('left_with_corners.jpg', img_left_with_corners)
    cv2.imwrite('right_with_corners.jpg', img_right_with_corners)

    # 构造投影矩阵
    P1 = np.hstack((K1, np.zeros((3, 1))))
    P2 = np.dot(K2, np.hstack((R, T)))

    # 三角测量
    points_4D = cv2.triangulatePoints(P1, P2, corners_left, corners_right)
    points_3D = points_4D[:3, :] / points_4D[3, :]

    # 获取四角和中心点的索引
    indices = [0, 6, 77, 84, 35]  # 棋盘格左上角、右上角、左下角、右下角、中心点

    # 将四角和中心的3D点结果标注在图像中
    for i in indices:
        x, y, z = points_3D[:, i]
        text = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f}"
        cv2.putText(img_left_with_corners, text, tuple(corners_left[i][0].astype(int) + (10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)

    # 保存带有3D点标注的图像
    cv2.imwrite('left_with_3D_points.jpg', img_left_with_corners)

    print("3D Points: \n", points_3D.T)
else:
    print("未能检测到棋盘格角点")
