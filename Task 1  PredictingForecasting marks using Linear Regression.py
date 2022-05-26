import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
# %matplotlib inline

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url)

data.head(5)

plt.axes().set_facecolor('xkcd:black')
plt.grid(color='#404040',linewidth=0.5)

plt.scatter(data["Hours"],data["Scores"],c='#fec615')

plt.title('Hours vs %',c='w')  
plt.xlabel('Hours',c='w')  
plt.ylabel('% Score',c='w')  
plt.show()

X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, 
                            test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train) 
print("Trained Algorithm usinf Linear Regression")

# Regression line Equation
line = lr.coef_*X+lr.intercept_

# line and scatter plot

plt.axes().set_facecolor('xkcd:black')
plt.grid(color='#404040',linewidth=0.5)
plt.scatter(X, y,c='#fec615')
plt.plot(X, line,c='#0485d1')
plt.show()

y_pred = lr.predict(X_test)
print("Prediction complete!")

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(comparison)

from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)

print('Mean Absolute Error:',mae,
      "\n\n",
      "Evaluation Complete!")

