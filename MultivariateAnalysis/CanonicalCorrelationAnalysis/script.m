clc,clear
%% һ.��������
filename = input('��һ������X������Դ�ļ���������׺��xlsx��txt������)��','s'); 
X = importdata(filename);
X = X.data;
filename = input('��һ������X������Դ�ļ���������׺��xlsx��txt������)��','s');
Y = importdata(fielname);
Y = Y.data; % �������'�����͵ı�����֧��ʹ�õ����������'Ҳ��Ӱ��

%% ��.���ϵ���������XX\XY\YX\YY�����ϵ��
XandY = [X Y];
R = corr(XandY); % �������'�����͵ı�����֧��ʹ�õ����������'Ҳ��Ӱ��

n1 = size(X,2);
n2 = size(Y,2);
s1 = R(1:n1,1:n1);
s12 = R(1:n1,(n1+1):end);
s21 = s12';
s2 = R(n1+1:end,n1+1:end);

%% ��. �������A��B��ʽ��1��
A = inv(s1)*s12*inv(s2)*s21;
B = inv(s2)*s21*inv(s1)*s12;

%% ��. ��A��B����������������ֵ�����Կ����������������ֵ��ʵ��һ���ģ�
[fv1,eigen1] = eig(A);% ��A����������feature vector������ֵeigenvalue
for i=1:n1
    fv1(:,i) = fv1(:,i)/sqrt(fv1(:,i)'*s1*fv1(:,i));    % ����������һ������Ҫ����x'*s1*x = 1(ʽ��2����
    fv1(:,i) = fv1(:,i)*sign(sum(fv1(:,i)));    % ��֤���з�����Ϊ��
end

% ��������ֵ��ƽ����
eigen1 = sqrt(diag(eigen1));
% �Ӵ�С����
[eigen1,ind1] = sort(eigen1,'descend');
% �����źõ�����ֵ��ѡ��Ӧ��ϵ�����X���ϵ����
a_x = fv1(:,ind1);
% �ѵ������ϵ��д��excel ��A1��ʼ�ķ���
filename = 'CCA.xlsx';
writematrix(a_x,filename,'Sheet',1,'Range','A1');
% �Ѷ�Ӧ������ֵд��excel 
writematrix(eigen1',filename,'Sheet',1,'Range',['A' num2str(n1+2)]);


% ��Y����ͬ����������
[fv2,eigen2] = eig(B);% ��B����������feature vector������ֵeigenvalue
for i=1:n2
    fv2(:,i) = fv2(:,i)/sqrt(fv2(:,i)'*s2*fv2(:,i));    % ����������һ������Ҫ����x'*s1*x = 1(ʽ��2����
    fv2(:,i) = fv2(:,i)*sign(sum(fv2(:,i)));    % ��֤���з�����Ϊ��
end

% ��������ֵ��ƽ����
eigen2 = sqrt(diag(eigen2));
% �Ӵ�С����
[eigen2,ind2] = sort(eigen2,'descend');
% �����źõ�����ֵ��ѡ��Ӧ��ϵ�����X���ϵ����
b_y = fv2(:,ind2);
% �ѵ������ϵ��д��excel ��A1��ʼ�ķ���
filename = 'CCA.xlsx';
writematrix(b_x,filename,'Sheet',1,'Range',['A' num2str(n1+4)]);
% �Ѷ�Ӧ������ֵд��excel 
writematrix(eigen2',filename,'Sheet',1,'Range',['A' num2str(n1+n2+6)]);
flag = n1+n2+6;   % �ļ���д����
%% ��. ԭʼ�����͵��ͱ���֮��������
xuR = s1*a_x;   % x��u�����ϵ��
yvR = s2*b_y;
xvR = s12*b_y;
yuR = s21*a_x;
% ����д��CCA�ļ���
flag = flag+2;
writematrix(xuR,filename,'Sheet',1,'Range',['A' num2str(flag)]);
flag=flag+1+n1;
writematrix(yvR,filename,'Sheet',1,'Range',['A' num2str(flag)]);
flag=flag+1+n2;
writematrix(xvR,filename,'Sheet',1,'Range',['A' num2str(flag)]);
flag=flag+1+n1;
writematrix(yuR,filename,'Sheet',1,'Range',['A' num2str(flag)]);

%% ��. ����ԭʼ���������ͱ������͵ķ���ı���
num = min(n1,n2);
mu = sum(xuR(:,1:num).^2)/n1;
mv = sum(yvR(:,1:num).^2)/n2;
fprintf('X���ԭʼ���ݱ�����u1-u%d���͵ı���Ϊ%f\n',num,sum(mu));
fprintf('Y���ԭʼ���ݱ�����v1-v%d���͵ı���Ϊ%f\n',num,sum(mv));

%% ��. �������ϵ���ļ���
