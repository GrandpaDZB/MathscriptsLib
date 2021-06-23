clc,clear
%% 一.导入数据
% txt、excel均可
A = importdata('anli10_1.txt');
% 若非纯数据则执行以下内容会自动提取数据矩阵
% A=A.data;
%% 二.R型聚类分析（指标聚类）(注：默认样本为行向量)
B = zscore(A);  % 数据标准化
R = corrcoef(B); % 计算相关系数矩阵
d = pdist(B','correlation'); % 计算相关系数导出的距离
z = linkage(d,'average'); % 按类平均法聚类
h = dendrogram(z);  % 画聚类图

% 然后根据聚类图选择合适的指标进行样本聚类
%% 三.Q型聚类分析（样本聚类）
% A为已删去不参与分析的指标矩阵（注：样本默认为行向量）
%A=A(:,[1 2 7 8 9 10]);
A = zscore(A); % 数据标准化
D = pdist(A); % 样本间的欧氏空间距离，每行使一个样本
Z = linkage(D,'average'); % 按类平均法聚类
H = dendrogram(Z); % 画聚类图
%% 四.结果处理
k = 3; % 划分为几类
T = cluster(Z,'maxclust',k);
for i=1:k
    tm = find(T==i);
    tm = reshape(tm,1,length(tm));
    fprintf('第%d类的有: %s\n',i,int2str(tm));
end;