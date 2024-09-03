import cv2
import numpy as np

# 加载图像
left_image = cv2.imread('left.bmp', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right.bmp', cv2.IMREAD_GRAYSCALE)

# 定义相机内参矩阵
K_left = np.array([[1399.80095010354, 0, 960.061069017239],
                   [0, 1400.36955196589, 530.408445039349],
                   [0, 0, 1]])

K_right = np.array([[1379.83293486266, 0, 947.508917171582],
                    [0, 1379.22317682187, 534.623140772307],
                    [0, 0, 1]])

# 使用ORB特征检测和匹配
orb = cv2.ORB_create()
keypoints_left, descriptors_left = orb.detectAndCompute(left_image, None)
keypoints_right, descriptors_right = orb.detectAndCompute(right_image, None)

# 使用BFMatcher进行特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_left, descriptors_right)
matches = sorted(matches, key=lambda x: x.distance)

# 提取匹配的点
pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches])
pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches])

# 计算基本矩阵 F
F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)

# 计算矫正变换
h1, w1 = left_image.shape[:2]
h2, w2 = right_image.shape[:2]

_, H1, H2 = cv2.stereoRectifyUncalibrated(pts_left, pts_right, F, imgSize=(w1, h1))

# 应用几何变换以矫正图像
left_rectified = cv2.warpPerspective(left_image, H1, (w1, h1))
right_rectified = cv2.warpPerspective(right_image, H2, (w2, h2))

# 保存矫正后的图像
cv2.imwrite('left_rectified.png', left_rectified)
cv2.imwrite('right_rectified.png', right_rectified)

# 创建SGBM对象用于计算视差图
window_size = 5
min_disp = 0  # 调整为0
num_disp = 128  # 视差范围

stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               P1=8 * 3 * window_size**2,
                               P2=32 * 3 * window_size**2,
                               disp12MaxDiff=1,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               preFilterCap=63)

# 计算视差图
disparity_map = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0

# 过滤掉负值和极小值
disparity_map[disparity_map < 0] = 0

# 显示或保存视差图
cv2.imshow('Disparity Map', disparity_map)
cv2.imwrite('disparity_map.png', disparity_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 检查视差图的最小值和最大值
print("Disparity Map min:", np.min(disparity_map))
print("Disparity Map max:", np.max(disparity_map))

# 将视差图转换为深度图
focal_length = K_left[0, 0]  # 使用左目相机的焦距
baseline = 120.0  # 基线长度为120mm

# 避免除以零
depth_map = np.zeros(disparity_map.shape)
valid_disp = disparity_map > 0
depth_map[valid_disp] = (focal_length * baseline) / disparity_map[valid_disp]
# 对深度图进行归一化处理，使其值范围在0到255之间
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_normalized = np.uint8(depth_map_normalized)

# 显示或保存归一化后的深度图
cv2.imshow('Normalized Depth Map', depth_map_normalized)
cv2.imwrite('normalized_depth_map.png', depth_map_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 可视化或保存深度图
cv2.imshow('Depth Map', depth_map)
cv2.imwrite('depth_map.png', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
