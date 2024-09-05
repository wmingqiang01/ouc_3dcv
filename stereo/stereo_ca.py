import numpy as np
import cv2

# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[1393.22172200895, -0.995545249279859, 955.251005896264],
                                         [0., 1393.64853177193, 533.727854721113],
                                         [0., 0., 1.]])
        # 右相机内参
        self.cam_matrix_right = np.array([[1392.85086761789, -1.11152594052554, 941.378478650626],
                                          [0., 1392.25346501003, 530.138196689747],
                                          [0., 0., 1.]])

        # 左右相机畸变系数：[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.167488181403578, 0.0436281136481580, -0.000295923284238189,
                                       -0.000557640732845627, -0.0447389245250197]])
        self.distortion_r = np.array([[-0.161597381187758, -0.0465705385781368, -0.000445502199937932,
                                       -0.000169101110520959, 0.332776709145332]])

        # 旋转矩阵
        self.R = np.array([[0.999962944723466,	0.00111956709473645,	0.00853555794866957],
                            [-0.00111382087540575,	0.999999149904264,	-0.000677933483298141],
                            [-0.00853630968464847,	0.000668401279658990,	0.999963341656432]])

        # 平移矩阵
        self.T = np.array([[-119.940339973533],	[0.0555434402027440],	[0.721611626472863]])

        # 主点列坐标的差
        self.doffs = 0.0

        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False

# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
def getRectifyTransform(height, width, config):
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    height = int(height)
    width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q

# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2

# 立体校正检验----画线
def draw_line(image1, image2):
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    line_interval = 150
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)
    return output

# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    img_channels = 1 if left_image.ndim == 2 else 3
    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 128,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.
    return trueDisp_left, trueDisp_right

# 根据公式计算深度图
def getDepthMapWithConfig(disparityMap, config):
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    depthMap = np.divide(fb, disparityMap + doffs)
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    reset_index2 = np.where(disparityMap < 0.0)
    depthMap[reset_index2] = 0
    return depthMap.astype(np.float32)


def normalize_depth_map(depthMap):
    # 排除无效的深度值（通常为0）来计算有效范围的最小值和最大值
    valid_depths = depthMap[depthMap > 0]
    min_depth = np.min(valid_depths)
    max_depth = np.max(valid_depths)

    # 归一化到0到1的范围
    normalized_depth_map = (depthMap - min_depth) / (max_depth - min_depth)

    # 将归一化的深度值缩放到0到255的范围
    depth_map_255 = (normalized_depth_map * 255).astype(np.uint8)

    # 将无效的深度值（0）保持不变
    depth_map_255[depthMap == 0] = 0

    return depth_map_255

if __name__ == "__main__":
    # 加载图像
    left_image = cv2.imread('left.jpg')
    right_image = cv2.imread('right.jpg')

    # 设置相机参数
    stereo_camera = stereoCamera()

    # 获取映射变换矩阵
    height, width = left_image.shape[:2]
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, stereo_camera)

    # 畸变校正和立体校正
    rectified_left, rectified_right = rectifyImage(left_image, right_image, map1x, map1y, map2x, map2y)

    # 立体校正检验----画线
    rectified_images_with_lines = draw_line(rectified_left, rectified_right)
    cv2.imwrite('rectified_images_with_lines.png', rectified_images_with_lines)

    # 计算视差图
    disparity_left, disparity_right = stereoMatchSGBM(rectified_left, rectified_right)
    cv2.imwrite('disparity_left.png', disparity_left)

    # 根据公式计算深度图
    depth_map_formula = getDepthMapWithConfig(disparity_left, stereo_camera)
    depth_map_formula = normalize_depth_map(depth_map_formula)
    cv2.imwrite('depth_map_formula.png', depth_map_formula)

    print("Processing complete. Outputs saved as images")
