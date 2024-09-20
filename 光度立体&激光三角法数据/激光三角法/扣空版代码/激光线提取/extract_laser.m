load ..\激光平面标定\cameraParams;

% 加载图像并进行去畸变
img = undistortImage(imread('D:\\Desktop\\光度立体&激光三角法数据\\激光三角法\\扣空版代码\\激光平面标定\\标定数据（20mm）\\24.png'), cameraParams);
img = rgb2gray(img);

% 获取图像尺寸，以及激光线大致区域
[h, w] = size(img);
crop_w = [845, 1040];  % 激光线的宽度范围
crop_h = [365, 846];   % 激光线的高度范围

% 用于显示记录激光线
maxImg = zeros(size(img));

% 用于记录激光点像素坐标
uv = [];

% 设置亮度阈值用于准确判断激光线
T = 200;

% 寻找每行的最大值并记录激光点的像素坐标
for i = crop_h(1):crop_h(2)
    % 在当前行的 crop_w 范围内寻找最大值
    row = img(i, crop_w(1):crop_w(2));  % 截取激光线所在区域
    [maxVal, maxIdx] = max(row);        % 找到该行中最大亮度值及其索引
    
    if maxVal > T
        % 如果最大值大于阈值，则认为是激光线点
        u = maxIdx + crop_w(1) - 1;     % 计算在原图中的 u 坐标
        v = i;                          % 行号就是 v 坐标
        
        % 记录激光点的 (u, v) 坐标
        uv = [uv; u, v];
        
        % 将找到的激光点在图像中标记
        maxImg(v, u) = 255;
    end
end

% 显示提取出的激光线
figure; imshow(maxImg); title('激光线提取');

% 保存提取激光线的像素坐标
save uv uv;
