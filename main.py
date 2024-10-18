import numpy as np
from bringingData import bring
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skfeature.function.similarity_based import fisher_score

from config import x_names

# initialize models and scaler
standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
linear_regression = LinearRegression()
lasso = Lasso(alpha=0.1)

# bring data
data = bring()

# code
scale_data = min_max_scaler.fit_transform(data)
x = scale_data[:, :5]
y = np.reshape(data[:, 5:], -1)
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.01, random_state=42)

# model = linear_regression.fit(x, y)

# weight_data = pd.DataFrame(zip(x_names, model.coef_))
# weight_data.columns = ['feature', 'coeff']
# weight_data = weight_data[weight_data.coeff.abs() > 0.01]
# weight_data = weight_data.sort_values('coeff', ascending=False)
# print(weight_data)
#
# sns.barplot(x='coeff', y='feature', data=weight_data)
# plt.title('Веса регрессии')
# plt.show()


# критерий фишера
# ranks = fisher_score.fisher_score(x, y)
#
# # график наших
# feature_importances = pd.Series(ranks, x_names)
# feature_importances.plot(kind='barh', color='teal')
# plt.show()


#корреляция Пирсона
x_df = pd.DataFrame(x, columns=['1','1','1','1','1'])
y_df = pd.DataFrame(y, columns=['1'])
pirson = x_df.corrwith(y_df, axis=0)
print(pirson)
