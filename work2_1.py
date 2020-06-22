import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

#糖尿病指數與血壓(blood pressure, bp)之一維線性廻歸. 需繪出分布圖與廻歸直線 

diabests = datasets.load_diabetes()#載入糖尿病資料庫

data = pd.DataFrame(diabests.data , columns =diabests.feature_names)#將data的內容變成dataframe的形式
target = pd.DataFrame(diabests.target)#將target的內容變成dataframe的形式

x = data['bp']#取出血壓資料
y = target[0]#糖尿病指數

xn = np.array(x).reshape(-1,1)#將Series轉成float64，size:(442,)->(442,1)
yn = np.array(y).reshape(-1,1)#將Series轉成float64，size:(442,)->(442,1)

#做線性回歸 
model = LinearRegression()
model.fit(xn,yn)
y1 = model.predict(xn)

#畫出圖形
plt.xlabel('Blood Pressure')
plt.ylabel('Diabests Index')
plt.plot(xn,y1,color='red',linewidth=3)
plt.scatter(x,y)