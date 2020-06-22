import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#糖尿病指數與血壓, BMI 之線性迴歸. 80% 為訓練資料, 20%為測試資料, 需輸出訓練資料與測試資料在廻歸分析中所得之分數

diabests = datasets.load_diabetes()#載入糖尿病資料庫

data = pd.DataFrame(diabests.data , columns =diabests.feature_names)#將data的內容變成dataframe的形式
target = pd.DataFrame(diabests.target)#


x =data[['bmi','bp']]#取出BMI,血壓資料
y= target[0]#取出糖尿病指數

xn = np.array(x).reshape(-1,2)#將Dataframe轉成float64，size:(442,2)->(4422)
yn = np.array(y).reshape(-1,1)#將Series轉成float64，size:(442,)->(442,1)

XTrain, XTest, YTrain, YTest = train_test_split(xn, yn, test_size=0.2,random_state=5)#80%訓練資料，20%測試資料

#線性回歸
model = LinearRegression() 
model.fit(XTrain ,YTrain )

#印出分數
print ( "訓練之分數=" , model.score( XTrain , YTrain))
print ( "測試之分數=" , model.score( XTest , YTest))