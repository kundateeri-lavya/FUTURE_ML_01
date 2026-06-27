import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv('data/sales_data.csv')
df['Day']=range(1,len(df)+1)
X=df[['Day']]
y=df['Sales']
model=LinearRegression()
model.fit(X,y)
future=pd.DataFrame({'Day':range(len(df)+1,len(df)+31)})
future['Predicted_Sales']=model.predict(future[['Day']])
future.to_csv('forecast.csv',index=False)
plt.plot(df['Day'],df['Sales'],label='Actual')
plt.plot(future['Day'],future['Predicted_Sales'],label='Forecast')
plt.legend()
plt.savefig('forecast_plot.png')
plt.show()
print(future.head())
