function temp = laser(p, inter, Rc_1, Tc_1)

% 输入 p 为 (u,v) 坐标，inter 为相机内参矩阵，Rc_1 为旋转矩阵，Tc_1 为平移向量

% 从相机内参中获取 fx, fy, cx, cy
fx = inter(1,1);
fy = inter(2,2);
cx = inter(1,3);
cy = inter(2,3);

u = p(1);
v = p(2);

% 归一化 x/z = (u - cx) / fx
normlization(1) = (u - cx) / fx;

% 归一化 y/z = (v - cy) / fy
normlization(2) = (v - cy) / fy;

% 旋转矩阵 R 的逆
R_inv = inv(Rc_1);

% 相机坐标系与世界坐标系有关系 [x , y , z]' = R * [X_w , Y_w , Z_w]' + T
% 对应 Z_w = 0 的情况，计算标定板的平面方程

r31 = R_inv(3,1);
r32 = R_inv(3,2);
r33 = R_inv(3,3);
t31 = Tc_1(1);
t32 = Tc_1(2);
t33 = Tc_1(3);

% 求平面方程的参数 a, b, c, d
a = r31;
b = r32;
c = r33;
d = r31 * t31 + r32 * t32 + r33 * t33;

% 利用归一化方程求解 z
z = d / (a * normlization(1) + b * normlization(2) + c);

% 求解 x 和 y
x = normlization(1) * z;
y = normlization(2) * z;

% 返回相机坐标系下的 [x, y, z]
temp = [x; y; z];

end
