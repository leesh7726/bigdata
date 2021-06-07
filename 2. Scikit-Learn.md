#  Scikit-Learn

#### Iris 분리 (특징행렬 X, 대상벡터 y)

1. seaborn

```
import seaborn as sns
iris = sns.load_dataset("iris")

X = iris.drop("species", axis=1)
y = iris["species"]
```

2. sklearn.datasets

```
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

print(X)
print(y)
```



### LinearRegression

#### 데이터 준비

```
import numpy as np

rs = np.random.RandomState(10)
x = 10 * rs.rand(100)
y = 3 * x +2 * rs.rand(100)
```



수치데이터를 이용하여 수치 데이터를 예측 -> 선형회귀 사용(LinearRegression)

```
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
```

```
from sklearn.linear_model import LinearRegression
regr = LinearRegression(fit_intercept=True)
```

* fit_intercept -> 인터셉트(절편)에 적합 시킬건지 여부 default = True



#### 특징행렬과 대상 벡터 준비

```
X = x.reshape(-1, 1)
X.shape, y.shape
```



#### 모델을 데이터에 적합

```
regr.fit(X, y)
regr.coef_ // 2.9855087
regr.intercept_ // 0.9878534341975644
```

- coef_ => 기울기
- intercept_ => 절편
- y = coef_X + intercept_
- y = 2.9855087x + 0.9878534341975644



#### 새로운 데이터를 이용해 예측

```
x_new = np.linspace(-1, 11, num=100)
X_new = x_new.reshape(-1, 1)

y_pred = regr.predict(X_new)
```



#### 모델 평가

```
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(rmse)
```

평균 제곱근 오차 => 0에 가까울수록 정확함.

실제값(y)와 예측값(pred_y)의 차이의제곱



















