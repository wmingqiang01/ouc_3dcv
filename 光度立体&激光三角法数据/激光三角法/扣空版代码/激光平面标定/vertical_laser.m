clear;
load cameraParams cameraParams;

% ��ȡ�������
RotationM = cameraParams.RotationMatrices;
TranslationV = cameraParams.TranslationVectors;
intrisic = cameraParams.IntrinsicMatrix;

% ѡ�������������� p[] ��
point_num = 6;  
p = zeros(3,point_num); % p ����ͼ��ƽ��� (u, v) ����

Loc = [];

% matlab �õ����ڲξ�����Ҫת��Ϊ������ʽ
% | fx  0   cx |
% | 0   fy  cy |
% | 0   0   1  |
inter = intrisic';

% ��������R,T,һ����Ӧѡȡ������ͼ��
Rc_1 = RotationM(:,:,24)'; % ��20��ͼ������
Tc_1 = TranslationV(24,:);
Rc_2 = RotationM(:,:,24)'; % ��21��ͼ������
Tc_2 = TranslationV(24,:);

% ����ѡȡ��6���������������� (u, v)
p(:,1) = [1039; 474; 1];  % �����1�� (u, v)
p(:,2) = [1024; 475; 1];  % �����2�� (u, v)
p(:,3) = [1023; 476; 1];  % �����3�� (u, v)
p(:,4) = [1025; 477; 1];  % �����4�� (u, v)
p(:,5) = [1025; 480; 1];  % �����5�� (u, v)
p(:,6) = [1025; 478; 1];  % �����6�� (u, v)

% ��������ת��Ϊ�������ϵ (x, y, z)
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

%% ����ƽ����� a * x + b * y + c * z + d = 0
X = Loc(1,:);  % ������ x ����
Y = Loc(2,:);  % ������ y ����
Z = Loc(3,:);  % ������ z ����

xyz = [ones(point_num,1) X' Y'];

% regress ��Ԫ���Իع� Z' = [ para(1) para(2) para(3) ] * xyz
para = regress(Z', xyz);

% para(2) * x + para(3) * y - z + para(1) = 0 ����ƽ��
planeParams_vertical = [para(2), para(3), -1, para(1)];
save planeParams_vertical planeParams_vertical;

%% ��ʾ��ϼ���ƽ��

xfit = min(X):1:max(X);  
yfit = min(Y):1:max(Y);

% [X,Y] = meshgrid(x,y) �������� x �� y �а��������귵�ض�ά�������ꡣ
[XFIT,YFIT] = meshgrid(xfit,yfit); 
ZFIT = para(1) + para(2) * XFIT + para(3) * YFIT;

% ��ʾ�����
figure, title('����ƽ��'), plot3(X, Y, Z, 'o'); 
axis equal;

% ��ʾ����ƽ��
hold on;
mesh(XFIT, YFIT, ZFIT);
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on;
hold off;
