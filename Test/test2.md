##### 1.

import numpy as np

X = np.array([7.71320643, 0.20751949, 6.33648235, 7.48803883, 4.98507012, 2.24796646, 1.98062865, 7.60530712, 1.69110837, 0.88339814])


y =np.array([24.51033893, 2.52934517, 19.01734358, 23.488501, 16.58045229,  7.9689515,  7.38539658, 23.3996735, 6.90887334, 4.07934599])

다음 x, y데이터를 사용한 선형회귀모델을 만들고

x_new = np.linspace(-1, 11, num=10)

일때 x_new에 대한 예측값 y_pred를 구하시오.



#### 2.

위의 y 와 y_pred를 평균 제곱근 오차를 사용하여 평가하시오.



#### 3.

iris 훈련 데이터와 예측 데이터 비율을 6:4로 나누어 knn 분류를 진행(n_neigbor=1로 설정)하여라.

