import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

df = pd.read_csv("carprices.csv")
print(df)
dummy=pd.get_dummies(df.CarModel)
merge=pd.concat([df,dummy],axis='columns')
final =merge.drop(['CarModel','Mercedez Benz C class'],axis='columns')
print(final)
x=final.drop(['Sell Price($)'],axis='columns')

y=final['Sell Price($)']

model=LinearRegression()
model.fit(x,y)
print(model.predict([[45000,4,0,0]]))
print(model.predict([[86000,7,0,1]]))
print(model.score(x,y)*100)