clc,clear
%% һ.��������
% txt��excel����
A = importdata('anli10_1.txt');
% ���Ǵ�������ִ���������ݻ��Զ���ȡ���ݾ���
% A=A.data;
%% ��.R�;��������ָ����ࣩ(ע��Ĭ������Ϊ������)
B = zscore(A);  % ���ݱ�׼��
R = corrcoef(B); % �������ϵ������
d = pdist(B','correlation'); % �������ϵ�������ľ���
z = linkage(d,'average'); % ����ƽ��������
h = dendrogram(z);  % ������ͼ

% Ȼ����ݾ���ͼѡ����ʵ�ָ�������������
%% ��.Q�;���������������ࣩ
% AΪ��ɾȥ�����������ָ�����ע������Ĭ��Ϊ��������
%A=A(:,[1 2 7 8 9 10]);
A = zscore(A); % ���ݱ�׼��
D = pdist(A); % �������ŷ�Ͽռ���룬ÿ��ʹһ������
Z = linkage(D,'average'); % ����ƽ��������
H = dendrogram(Z); % ������ͼ
%% ��.�������
k = 3; % ����Ϊ����
T = cluster(Z,'maxclust',k);
for i=1:k
    tm = find(T==i);
    tm = reshape(tm,1,length(tm));
    fprintf('��%d�����: %s\n',i,int2str(tm));
end;