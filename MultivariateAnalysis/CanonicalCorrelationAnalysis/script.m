clc,clear
%% 一.加载数据
filename = input('第一组向量X数据来源文件名（带后缀，xlsx或txt都可以)：','s'); 
X = importdata(filename);
X = X.data;
filename = input('第一组向量X数据来源文件名（带后缀，xlsx或txt都可以)：','s');
Y = importdata(fielname);
Y = Y.data; % 如果报错'此类型的变量不支持使用点进行索引。'也不影响

%% 二.相关系数矩阵并提出XX\XY\YX\YY的相关系数
XandY = [X Y];
R = corr(XandY); % 如果报错'此类型的变量不支持使用点进行索引。'也不影响

n1 = size(X,2);
n2 = size(Y,2);
s1 = R(1:n1,1:n1);
s12 = R(1:n1,(n1+1):end);
s21 = s12';
s2 = R(n1+1:end,n1+1:end);

%% 三. 计算矩阵A、B，式（1）
A = inv(s1)*s12*inv(s2)*s21;
B = inv(s2)*s21*inv(s1)*s12;

%% 四. 求A、B的特征向量和特征值（可以看出两个矩阵的特征值其实是一样的）
[fv1,eigen1] = eig(A);% 求A的特征向量feature vector和特征值eigenvalue
for i=1:n1
    fv1(:,i) = fv1(:,i)/sqrt(fv1(:,i)'*s1*fv1(:,i));    % 特征向量归一化，且要满足x'*s1*x = 1(式（2））
    fv1(:,i) = fv1(:,i)*sign(sum(fv1(:,i)));    % 保证所有分量和为正
end

% 计算特征值的平方根
eigen1 = sqrt(diag(eigen1));
% 从大到小排列
[eigen1,ind1] = sort(eigen1,'descend');
% 根据排好的特征值挑选对应的系数提出X组的系数阵
a_x = fv1(:,ind1);
% 把典型相关系数写入excel 从A1开始的方块
filename = 'CCA.xlsx';
writematrix(a_x,filename,'Sheet',1,'Range','A1');
% 把对应的特征值写入excel 
writematrix(eigen1',filename,'Sheet',1,'Range',['A' num2str(n1+2)]);


% 对Y进行同样操作……
[fv2,eigen2] = eig(B);% 求B的特征向量feature vector和特征值eigenvalue
for i=1:n2
    fv2(:,i) = fv2(:,i)/sqrt(fv2(:,i)'*s2*fv2(:,i));    % 特征向量归一化，且要满足x'*s1*x = 1(式（2））
    fv2(:,i) = fv2(:,i)*sign(sum(fv2(:,i)));    % 保证所有分量和为正
end

% 计算特征值的平方根
eigen2 = sqrt(diag(eigen2));
% 从大到小排列
[eigen2,ind2] = sort(eigen2,'descend');
% 根据排好的特征值挑选对应的系数提出X组的系数阵
b_y = fv2(:,ind2);
% 把典型相关系数写入excel 从A1开始的方块
filename = 'CCA.xlsx';
writematrix(b_x,filename,'Sheet',1,'Range',['A' num2str(n1+4)]);
% 把对应的特征值写入excel 
writematrix(eigen2',filename,'Sheet',1,'Range',['A' num2str(n1+n2+6)]);
flag = n1+n2+6;   % 文件的写入行
%% 五. 原始变量和典型变量之间的相关性
xuR = s1*a_x;   % x和u的相关系数
yvR = s2*b_y;
xvR = s12*b_y;
yuR = s21*a_x;
% 依次写入CCA文件中
flag = flag+2;
writematrix(xuR,filename,'Sheet',1,'Range',['A' num2str(flag)]);
flag=flag+1+n1;
writematrix(yvR,filename,'Sheet',1,'Range',['A' num2str(flag)]);
flag=flag+1+n2;
writematrix(xvR,filename,'Sheet',1,'Range',['A' num2str(flag)]);
flag=flag+1+n1;
writematrix(yuR,filename,'Sheet',1,'Range',['A' num2str(flag)]);

%% 六. 各组原始变量被典型变量解释的方差的比例
num = min(n1,n2);
mu = sum(xuR(:,1:num).^2)/n1;
mv = sum(yvR(:,1:num).^2)/n2;
fprintf('X组的原始数据被变量u1-u%d解释的比例为%f\n',num,sum(mu));
fprintf('Y组的原始数据被变量v1-v%d解释的比例为%f\n',num,sum(mv));

%% 七. 典型相关系数的检验
