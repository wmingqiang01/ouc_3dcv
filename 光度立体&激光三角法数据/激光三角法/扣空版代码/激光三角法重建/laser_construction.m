load ..\激光线提取\uv.mat
load ..\激光平面标定\cameraParams.mat
load ..\激光平面标定\planeParams_vertical

% 图像归一化坐标
inter = cameraParams.IntrinsicMatrix';

u = uv(:,1);
v = uv(:,2);

norm_x = (u - inter(1,3)) / inter(1,1);  % 归一化 x/z = (u - cx) / fx
norm_y = (v - inter(2,3)) / inter(2,2);  % 归一化 y/z = (v - cy) / fy

% 加载激光平面方程，计算相机坐标系下的激光线
planeParams = planeParams_vertical;
pa = planeParams(1);
pb = planeParams(2);
pc = planeParams(3);
pd = planeParams(4);

% 计算相机坐标系下的 z 值
z = -pd ./ (pa .* norm_x + pb .* norm_y + pc);

% 计算相机坐标系下的 x, y 坐标
x = norm_x .* z;
y = norm_y .* z;

% 将相机坐标系下的激光点变换到棋盘格平面世界坐标系
% 外参：旋转矩阵和平移向量（选择合适的相机位姿）
Rc = cameraParams.RotationMatrices(:,:,1);  % 假设使用第一张图像的外参
Tc = cameraParams.TranslationVectors(1,:);  % 对应第一张图像的平移向量

% 将点从相机坐标系转换到世界坐标系
% 世界坐标系点: [X_w, Y_w, Z_w]' = inv(R) * ([x, y, z]' - T)
% Tc 是行向量，需要转置操作
worldPoints = Rc' * ([x'; y'; z'] - Tc');

% 提取世界坐标系下的 X_w, Y_w, Z_w
X_w = worldPoints(1, :);
Y_w = worldPoints(2, :);
Z_w = worldPoints(3, :);

% 显示激光点在世界坐标系下的分布
figure;
scatter3(X_w, Y_w, Z_w, 'filled');
title('激光点在世界坐标系下的分布');
xlabel('X (世界坐标系)');
ylabel('Y (世界坐标系)');
zlabel('Z (世界坐标系)');
grid on;
axis equal;

