%***
%PS�ؽ�--matlab�ٷ��汾
%writen by FanHao
%date 2016.4.16

clear all;close all;clc;
addpath(genpath(pwd));  % ��ӵ�ǰ·���µ�������Ŀ¼
g_pic_num = 6;
g_src = 'D:\\Desktop\\�������&�������Ƿ�����\\�������\\pictures\\';
shadowThresh = 0.1;
%% ����ͼƬ
image = imread([g_src '0.bmp']);
image = image(:,:,1);
[rows, cols,c] = size(image);
image = image(rows*1/8+1:rows*7/8, cols*1/8+1:cols*7/8);
    
[M, N, C] = size(image);
I = ones(M, N, g_pic_num);
for  i = 1:g_pic_num
    name = (i-1) * 360/g_pic_num;
    image = im2double(imread([g_src [int2str(name) '.bmp']]));
    image = image(:,:,1);
%     image = rgb2gray(image);
    [rows, cols] = size(image);
    image = image(rows*1/8+1:rows*7/8, cols*1/8+1:cols*7/8);
    I(:,:, i) = image /prctile(image(:), 99); %%�о����ǿ���ų��߹�����˼
%     imshow(image);
%     pause; 
end
clear name rows cols
%% ͼ��ĳ߶���Ϣ
[g_rows, g_cols] = size(image);
g_length = g_rows * g_cols;

%% ������սǶ�
L = zeros(3, g_pic_num);
Slant = 45;
Slant_sin = sin(Slant/180*pi);
Slant_cos = cos(Slant/180*pi);
for i = 1:g_pic_num
    Tilt = (i-1) * 360/g_pic_num;
    %%��ʵ�������ϵ
    L(1,i) = Slant_sin * cos(Tilt/180*pi);
    L(2,i) = Slant_sin * sin(Tilt/180*pi);
    L(3,i) = Slant_cos;
end
clear Slant��Slant_sin Slant_cos i

% Create a shadow mask.
shadow_mask = (I > shadowThresh);
se = strel('disk', 2);
for i = 1:g_pic_num
  % Erode the shadow map to handle de-mosaiking artifact near shadow boundary.
  shadow_mask(:,:,i) = imerode(shadow_mask(:,:,i), se);
end

[rho, n] = PhotometricStereo(I, shadow_mask, L); %������庯�����

%% Visualize the normal map. axis xy;
N_RGB(:,:,1)  = (n(:,:,1) + 1) / 2;
N_RGB(:,:,2)  = (n(:,:,2) + 1) / 2;
N_RGB(:,:,3)  = n(:,:,3);
figure; imshow(N_RGB); 
% figure; imshow(rho);

%% Estimate depth map from the normal vectors.
fprintf('Estimating depth map from normal vectors...\n');
p = -n(:,:,1) ./ n(:,:,3);
q = -n(:,:,2) ./ n(:,:,3);
p(isnan(p)) = 0; %�ж������Ԫ���Ƿ���NaN�� NaN �� Not a Number ����д��
q(isnan(q)) = 0;

% figure; subplot(1,2,1); mesh(p);
% subplot(1,2,2); mesh(q);
%% integration
% mask = ones(g_rows, g_cols);
% Height_i = integrate_horn2(p, q, mask, 5000, 1);
% Height_i = flipud(Height_i);
% figure;mesh(Height_i);

% Height_poisson =poisson_solver_function_neumann(p, q);
% Height_poisson = flipud(Height_poisson);
% figure;mesh(Height_poisson);

Z = DepthFromGradient(p, q);
Z(isnan(n(:,:,1)) | isnan(n(:,:,2)) | isnan(n(:,:,3))) = NaN;
Z = flipud(Z);

figure;surf(Z, 'EdgeColor', 'None', 'FaceColor', [0.5 0.5 0.5]);
axis equal; camlight;view(-75, 30);

figure;mesh(Z);axis equal; camlight;view(-75, 30);
