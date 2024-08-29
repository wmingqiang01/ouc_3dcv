# -*- coding = utf-8 -*-
# @Time : 2024/8/28 20:25
# @Author : 杨沛霖
# @File : testOut.py
# @Software : PyCharm

import cv2
import numpy as np
import glob
import math

# 设置棋盘格参数
chessboard_size = (8,11)  # 棋盘格的内角点个数
square_size = 3.0  # 棋盘格每个方格的实际大小，单位可以是毫米、厘米或米

# 准备棋盘格的世界坐标系坐标（假设z=0）
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
# item = 0
# for i in range(0,chessboard_size[0]):
#     for j in range(0,chessboard_size[1]):
#         objp[item,0] = i
#         objp[item,1] = j
#         item += 1
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size



# 用于存储所有图像的世界坐标系点和图像坐标系点
objpoints = []  # 世界坐标系中的点
imgpoints = []  # 图像坐标系中的点

# 获取所有棋盘格图像的路径
images = glob.glob('./img3/*.jpg')

for image_path in images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)




    if ret:
        start = corners[0][0]
        objpoints.append(objp)
        imgpoints.append(corners)
        # 绘制角点以进行可视化
        #img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        # img = cv2.drawMarker(img,(int(start[0]),int(start[1])),color=(255, 255, 0),thickness=2,markerType=cv2.MARKER_STAR,line_type=cv2.LINE_8,markerSize=20)
        # #img_small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        # cv2.imshow('Chessboard Corners',img)
        cv2.waitKey(0)



cv2.destroyAllWindows()


# 执行相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
# 输出相机的内参和畸变系数
# print("Camera matrix:\n", mtx)
# print("Distortion coefficients:\n", dist)

# 保存内参和畸变系数到文件
np.savez('camera_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


Camera_intrinsic = {"mtx": mtx, "dist": dist }

for image_path in images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:

        lam = -1*  (corners[1][0][0] - corners[0][0][0])/3.0

        _, rvec, tvec = cv2.solvePnP(objp, corners, Camera_intrinsic["mtx"], Camera_intrinsic["dist"])  # 解算位姿
        distance = math.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2)  # 计算距离
        rvec_matrix = cv2.Rodrigues(rvec)[0]  # 旋转向量->旋转矩阵

        # print("旋转矩阵")
        # print(rvec_matrix)
        # print("tvec")
        # print(tvec)
        # print("距离")
        # print(distance)
        print(tvec)
        rt = np.array([[rvec_matrix[0][0], rvec_matrix[0][1],rvec_matrix[0][2]],
                       [rvec_matrix[1][0], rvec_matrix[1][1],rvec_matrix[1][2]],
                       [rvec_matrix[2][0], rvec_matrix[2][1],rvec_matrix[2][2]],
                       ], dtype=np.float)
        rt_i = np.linalg.inv(rt)
        # pi_i = np.linalg.inv(Camera_intrinsic["mtx"])
        #
        # ax = Camera_intrinsic["mtx"][0][2]
        # ay = Camera_intrinsic["mtx"][1][2]
        #
        # # print(Camera_intrinsic)
        #
        #
        # uv = np.array([[ax],
        #                [ay],
        #                [1.0]])
        #
        #
        ti_i = np.array([[tvec[0][0]],
                       [tvec[1][0]],
                       [tvec[2][0]]])
        #
        # # print("tvec")
        # # print(tvec)
        #
        # print("pi_i.dot(uv)")
        # print(pi_i.dot(uv))
        #
        xy1 = rt_i.dot(ti_i)
        print("xy1")
        print(xy1)

        # rt = np.array([[rvec_matrix[0][0], rvec_matrix[0][1], tvec[0]],
        #                [rvec_matrix[1][0], rvec_matrix[1][1], tvec[1]],
        #                [rvec_matrix[2][0], rvec_matrix[2][1], tvec[2]]], dtype=np.float)
        # rt_i = np.linalg.inv(rt)
        # pi_i = np.linalg.inv(Camera_intrinsic["mtx"])
        #
        #
        #
        # uv = np.array([[ax],
        #                [ay],
        #                [1.]])
        # xy1 =  rt_i.dot(pi_i.dot(uv)) / lam
        # print("xy1")
        # print(xy1)

        # 绘制角点以进行可视化
        # img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        # img_small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        #cv2.imshow('Chessboard Corners',img_small)
        cv2.waitKey(0)



cv2.destroyAllWindows()

