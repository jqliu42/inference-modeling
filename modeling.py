import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.externals import joblib

# 使用收集的数据进行建模

def mape(y_true,y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true)) * 100

np.random.seed(42)

filename = 'fc_data.txt'
#filename = 'conv_data.txt'

data = np.loadtxt(filename,delimiter='\t')

Y = data[:,-1]
#print(np.max(Y),np.min(Y),np.mean(Y),np.median(Y))

xtrain, xtest,ytrain,ytest = train_test_split(data[:,0:-1],data[:,-1],test_size=0.2)

std = StandardScaler()
xtrain = std.fit_transform(xtrain)
xtest = std.fit_transform(xtest)

#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(xtrain,ytrain)

pred = model.predict(xtest)
rmse = np.sqrt(mean_squared_error(pred,ytest))
mape = mape(ytest,pred)

print("result: ",lin_rmse,", mape:",lin_mape)

plt.scatter(np.log(pred),np.log(ytest))
plt.xlabel("prediction")
plt.ylabel("groud truth")

joblib.dump(forest_reg,"forest_reg_conv.pkl")
