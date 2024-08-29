import cv2
import numpy as np
import glob

# 加载数据
with np.load('data.npz') as X:
    # 加载上一部生成的参数
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
    # criteria:角点精准化迭代过程的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 棋盘格模板规格
len = 18  # 黑白格长度
w = 6
h = 13


# 画坐标轴和立方体
def draw(img, corners, imgpts, imgpts2):
    corner = tuple(map(int, corners[0].ravel()))  # 确保corner是整数类型的元组
    # img要画的图像，corner起点，tuple终点，颜色，粗细
    img = cv2.line(img, corner, tuple(map(int, imgpts2[0].ravel())), (255, 0, 0), 8)
    img = cv2.line(img, corner, tuple(map(int, imgpts2[1].ravel())), (0, 255, 0), 8)
    img = cv2.line(img, corner, tuple(map(int, imgpts2[2].ravel())), (0, 0, 255), 8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'X', tuple(map(int, imgpts2[0].ravel() + 2)), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Y', tuple(map(int, imgpts2[1].ravel() + 2)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Z', tuple(map(int, imgpts2[2].ravel() + 2)), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    imgpts = np.int32(imgpts).reshape(-1, 2)  # 将imgpts转换为整数
    for i, j in zip(range(4), range(4, 8)):  # 正方体顶点逐个连接
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 215, 0), 3)  # 画上层正方形的边

    img = cv2.drawContours(img, [imgpts[4:]], -1, (255, 215, 0), 3)  # 画上面的盖子
    img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 215, 0), 3)  # 画下面的盖子
    return img


objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w * len:len, 0:h * len:len].T.reshape(-1, 2)
axis = np.float32([[0, 0, 0], [0, 2 * len, 0], [2 * len, 2 * len, 0], [2 * len, 0, 0],
                   [0, 0, -2 * len], [0, 2 * len, -2 * len], [2 * len, 2 * len, -2 * len], [2 * len, 0, -2 * len]])
axis2 = np.float32([[3 * len, 0, 0], [0, 3 * len, 0], [0, 0, -3 * len]]).reshape(-1, 3)
images = glob.glob('caliimgs/*.jpg')
i = 1;
for fname in images:

    img = cv2.imread(fname)
    img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    # 寻找角点，存入corners，ret是找到角点的flag
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret is True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 求解物体位姿的需要
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        # projectPoints()根据所给的3D坐标和已知的几何变换来求解投影后的2D坐标。
        # imgpts是整体的8个顶点
        imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # imgpts2是三个坐标轴的x,y,z划线终点
        imgpts2, _ = cv2.projectPoints(axis2, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts, imgpts2)
        cv2.imshow('世界坐标系与小盒子', img)
        cv2.imwrite('imgs/'+str(i) + '.png', img)
        cv2.waitKey(1000)
        i += 1;

cv2.destroyAllWindows()
print("完毕")

