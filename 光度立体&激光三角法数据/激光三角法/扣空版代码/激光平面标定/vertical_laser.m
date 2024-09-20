clear;
load cameraParams cameraParams;

% 获取内外参数
RotationM = cameraParams.RotationMatrices;
TranslationV = cameraParams.TranslationVectors;
intrisic = cameraParams.IntrinsicMatrix;

% 选择六个激光点存入 p[] 中
point_num = 6;  
p = zeros(3,point_num); % p 代表图像平面的 (u, v) 坐标

Loc = [];

% matlab 得到的内参矩阵需要转置为常见形式
% | fx  0   cx |
% | 0   fy  cy |
% | 0   0   1  |
inter = intrisic';

% 这里的外参R,T,一定对应选取激光点的图像
Rc_1 = RotationM(:,:,24)'; % 第20张图像的外参
Tc_1 = TranslationV(24,:);
Rc_2 = RotationM(:,:,24)'; % 第21张图像的外参
Tc_2 = TranslationV(24,:);

% 填入选取的6个激光点的像素坐标 (u, v)
p(:,1) = [1039; 474; 1];  % 激光点1的 (u, v)
p(:,2) = [1024; 475; 1];  % 激光点2的 (u, v)
p(:,3) = [1023; 476; 1];  % 激光点3的 (u, v)
p(:,4) = [1025; 477; 1];  % 激光点4的 (u, v)
p(:,5) = [1025; 480; 1];  % 激光点5的 (u, v)
p(:,6) = [1025; 478; 1];  % 激光点6的 (u, v)

% 逐个激光点转换为相机坐标系 (x, y, z)
temp = laser(p(:,1), inter, Rc_1, Tc_1);
Loc = [Loc temp];

temp = laser(p(:,2), inter, Rc_1, Tc_1);
Loc = [Loc temp];

temp = laser(p(:,3), inter, Rc_1, Tc_1);
Loc = [Loc temp];

temp = laser(p(:,4), inter, Rc_2, Tc_2);
Loc = [Loc temp];

temp = laser(p(:,5), inter, Rc_2, Tc_2);
Loc = [Loc temp];

temp = laser(p(:,6), inter, Rc_2, Tc_2);
Loc = [Loc temp];

%% 激光平面拟合 a * x + b * y + c * z + d = 0
X = Loc(1,:);  % 激光点的 x 坐标
Y = Loc(2,:);  % 激光点的 y 坐标
Z = Loc(3,:);  % 激光点的 z 坐标

xyz = [ones(point_num,1) X' Y'];

% regress 多元线性回归 Z' = [ para(1) para(2) para(3) ] * xyz
para = regress(Z', xyz);

% para(2) * x + para(3) * y - z + para(1) = 0 激光平面
planeParams_vertical = [para(2), para(3), -1, para(1)];
save planeParams_vertical planeParams_vertical;

%% 显示拟合激光平面

xfit = min(X):1:max(X);  
yfit = min(Y):1:max(Y);

% [X,Y] = meshgrid(x,y) 基于向量 x 和 y 中包含的坐标返回二维网格坐标。
[XFIT,YFIT] = meshgrid(xfit,yfit); 
ZFIT = para(1) + para(2) * XFIT + para(3) * YFIT;

% 显示激光点
figure, title('激光平面'), plot3(X, Y, Z, 'o'); 
axis equal;

% 显示激光平面
hold on;
mesh(XFIT, YFIT, ZFIT);
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on;
hold off;
