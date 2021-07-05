# Logistic Regression

## 1 Theory

Consider a linear regression without bias
$$
\hat y=f(x)=w^{T}x
$$
if we expect the range of output value to be in the section (0,1), a sigmoid map can help. We do a transform like below
$$
\hat y = sigmoid(f(x))=sigmoid(w^{T}x)
$$
Then it becomes a simple optimization problem. We choose cross entropy  as our Loss function because norm_2 manifests a lower learning speed. As you can imagine, an exp function leads to a very small gradient.
$$
L = \Pi_i f(x_i)^{y_i}(1-f(x_i))^{1-y_i} \\
lnL=\Sigma_i y_ilnf(x_i)+(1-y_i)ln(1-f(x_i)) \\
\partial lnL/\partial w=\Sigma_i (f(x_i)-y_i)x_i
$$
We use gradient descend method to magnify the maximum likelihood function.
$$
w^{new}=w^{old} + \epsilon (\partial lnL/\partial w )
$$
It 's easy to work of course. However, logistic regression can only work on a linear classification problem and it's also easy to over fit training data. For a nonlinear or inseparable problem you can try Support Vector Machine.

## 2 How to use

first of all, instantiate the class 'Logistic Regression'

```python
from LogisticRegression import LogisticRegression

# instantiation
model = LogisticRegression()
```

function data_loader helps to import data  from an excel sheet, which is undertaken by package "xlrd", so only the "xls" format is supported.

In the data sheet, please be sure about the training data is in form like
$$
sheet=
\left[
\begin{matrix}
x_1 & y_1 \\
x_2 & y_2 \\
\vdots & \vdots \\
x_n & y_n
\end{matrix}
\right]
$$

```python
file = './test_data.xls'
model.data_loader(file)
```

function fit helps to train the model. if you run data_loader first, you don't need to give the training data. And if you want to use a new set of training data, you can add them in the parameters X/Y(default: X = [], Y = []).  Also max iteration and learning rate can be modified here.

```python
model.fit(iter = 10000, step = 0.1)
```

```python
model.predict(x)
```

