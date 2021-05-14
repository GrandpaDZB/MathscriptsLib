clear;
clc;
expect_dim = input('expect_dim = ');
X = input('X = ');
X_size = size(X);
Y = X*X';
[eigen_vectors, eigen_values] = eig(Y);

%sort
for i = 1:X_size(1)
    min = i;
    for j = i+1:X_size(1)
        if eigen_values(j,j) < eigen_values(min, min)
            min = j;
        end
    end
    tmp_value = eigen_values(min, min);
    tmp_vector = eigen_vectors(:, min);
    eigen_values(min, min) = eigen_values(i, i);
    eigen_vectors(:, min) = eigen_vectors(:, i);
    eigen_values(i, i) = tmp_value;
    eigen_vectors(:, i) = tmp_vector;
end


D = rand(X_size(1), expect_dim);
for i=1:expect_dim
    D(:,i) = eigen_vectors(:, end - i + 1);
end

C = D'*X;