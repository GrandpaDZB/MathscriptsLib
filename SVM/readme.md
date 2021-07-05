# Support Vector Machine

## 1 原理

## 2 如何使用

```python
from SVM import SVM

model = SVM( 
    max_iter = 10000,
    kernal_type = 'linear',
    C = 1.0,
    epsilon = 0.001,
    Gaussian_stderr = 1.0
)
# max_iter default: 10000
# kernal_type the kernal you want to use, here you can only choose linear or Gaussian  default: linear
# C Relaxation Factor, an infinity C indicate a linear  separable system  default: 1.0
# epsilon Learning rate  default: 0.001
# Gaussian_stderr  GaussianKernal parameter, modify this parameters to train the model with a better fitting result. Easy to be over fitting when it's too big, and easy to be underfitting when it's too small.

file = "./data.xls"
(X_train, Y_train) = model.data_loader(file, sheet_index = 0)
# sheet_index and area are not necessary

model.fit(X_train, Y_train)
model.predict(x)
```

我相信MATLAB原生的SVM库会有更快的计算速度和更好的效果