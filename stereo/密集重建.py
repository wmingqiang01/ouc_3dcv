import cv2
import numpy as np
import glob
import os

# Step 1: Camera Calibration
pattern_size = (11, 8)  # 内角点数量
square_size = 30  # 单元格边长，单位为mm

# 准备棋盘格的3D点（世界坐标系中的坐标）
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 存储所有的3D点（世界坐标系）和2D点（图像坐标系）
objpoints = []  # 3D点
imgpoints_left = []  # 左目图像的2D点
imgpoints_right = []  # 右目图像的2D点

# 读取左目和右目图像路径
left_image_path = r'D:\Desktop\双目数据\深度估计\left'
right_image_path = r'D:\Desktop\双目数据\标定\right'
left_images = sorted(glob.glob(left_image_path + '\*.bmp'))
right_images = sorted(glob.glob(right_image_path + '\*.bmp'))

# 遍历所有图像，查找棋盘格角点
for left_img_path, right_img_path in zip(left_images, right_images):
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

    # 如果找到足够的角点，添加到点列表中
    if ret_left and ret_right:
        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

cv2.destroyAllWindows()

# 左目相机标定
ret_left, camera_matrix_left, dist_coeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray_left.shape[::-1], None, None)

# 右目相机标定
ret_right, camera_matrix_right, dist_coeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

# 立体矫正
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    camera_matrix_left, dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right,
    gray_left.shape[::-1], criteria=criteria_stereo, flags=flags
)

# 立体矫正
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    camera_matrix_left, dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right,
    gray_left.shape[::-1], R, T, alpha=0
)

# 计算去畸变和矫正的映射
map1_left, map2_left = cv2.initUndistortRectifyMap(
    camera_matrix_left, dist_coeffs_left, R1, P1, gray_left.shape[::-1], cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    camera_matrix_right, dist_coeffs_right, R2, P2, gray_right.shape[::-1], cv2.CV_16SC2)

# 读取需要进行深度估计的图像
left_image_for_depth = 'left.jpg'
right_image_for_depth = 'right.jpg'
img_left = cv2.imread(left_image_for_depth, cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread(right_image_for_depth, cv2.IMREAD_GRAYSCALE)

# 图像矫正
rectified_img_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
rectified_img_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

# Step 3: Dense Matching and 3D Reconstruction
# 创建SGBM匹配器
window_size = 5
min_disp = 0
num_disp = 16 * 5  # 必须是16的倍数
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# 计算视差图
disparity = stereo.compute(rectified_img_left, rectified_img_right).astype(np.float32) / 16.0

# 保存视差图
cv2.imwrite('disparity_1.png', disparity)

# 三维重建
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# 可视化和保存3D点云
mask = disparity > disparity.min()  # 创建掩膜，仅保留有效的视差点
points = points_3D[mask]  # 筛选有效的3D点

# 使用颜色保存点云（可选）
colors = cv2.imread(left_image_for_depth)  # 从左目图像获取颜色信息
colors = colors[mask]

# 创建输出文件夹
output_folder = r'F:\abc\pointcloud'
os.makedirs(output_folder, exist_ok=True)

# 保存为PLY文件（可视化3D点云）
output_path = os.path.join(output_folder, 'cloud_1.ply')
with open(output_path, 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write(f'element vertex {len(points)}\n')
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('end_header\n')
    for p, c in zip(points, colors):
        f.write(f'{p[0]} {p[1]} {p[2]} {c[2]} {c[1]} {c[0]}\n')  # PLY文件使用RGB格式

print("深度估计和3D点云生成完成！")
