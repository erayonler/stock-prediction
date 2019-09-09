
## Part 0: Import Libraries


```python
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
%matplotlib inline
```

## Part 1: Importing the data


```python
facebook = pd.read_csv("static/FB.csv")
```


```python
facebook.set_index("Date", inplace=True)
```


```python
facebook.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-05-18</th>
      <td>42.049999</td>
      <td>45.000000</td>
      <td>38.000000</td>
      <td>38.230000</td>
      <td>38.230000</td>
      <td>573576400</td>
    </tr>
    <tr>
      <th>2012-05-21</th>
      <td>36.529999</td>
      <td>36.660000</td>
      <td>33.000000</td>
      <td>34.029999</td>
      <td>34.029999</td>
      <td>168192700</td>
    </tr>
    <tr>
      <th>2012-05-22</th>
      <td>32.610001</td>
      <td>33.590000</td>
      <td>30.940001</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>101786600</td>
    </tr>
    <tr>
      <th>2012-05-23</th>
      <td>31.370001</td>
      <td>32.500000</td>
      <td>31.360001</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>73600000</td>
    </tr>
    <tr>
      <th>2012-05-24</th>
      <td>32.950001</td>
      <td>33.209999</td>
      <td>31.770000</td>
      <td>33.029999</td>
      <td>33.029999</td>
      <td>50237200</td>
    </tr>
  </tbody>
</table>
</div>




```python
facebook["Adj Close"].plot(label="Facebook", figsize=(12,8),title="Adj Close Prices")
plt.legend();
```


![png](output_6_0.png)



```python
facebook["Volume"].plot(label="Facebook", figsize=(12,8),title="Volume")
plt.legend();
```


![png](output_7_0.png)


* Interesting, looks like Facebook has a spike somewhere in late 2012


```python
facebook["Volume"].max()
```




    573576400




```python
facebook["Volume"].argmax()
```




    '2012-05-18'



When we search about that date we saw it was the date of Facebook IPO - https://money.cnn.com/2012/05/18/technology/facebook-ipo-trading/index.htm

** Then I tried to create some more features. a "Total Traded" column which is Open Price multiplied by the Volume trade **


```python
facebook["Total Traded"] = facebook["Open"]*facebook["Volume"]
```


```python
facebook["Total Traded"].plot(label="Facebook", figsize=(16,8),title="Opening Prices")
plt.legend();
```


![png](output_14_0.png)


Something happened for facebook at 2018. There is spike for Total Trade


```python
facebook["Total Traded"].max()
```




    29696968923.196297




```python
facebook["Total Traded"].argmax()
```




    '2018-07-26'



the company missed projections on key metrics after struggling with data leaks and fake news scandals. - https://www.cnbc.com/2018/07/26/facebook-is-on-pace-for-its-worst-day-ever.html

Then I used MA (Moving Averages) to reduce to noise and see the trend better.


```python
facebook["MA30"] = facebook["Adj Close"].rolling(window=30).mean()
facebook["MA100"] = facebook["Adj Close"].rolling(window=100).mean()
facebook[["Adj Close", "MA30", "MA100"]].plot(figsize=(16,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1c98e5c0>




![png](output_20_1.png)


### Part 3: Basic Financial Analysis

#### Daily Percentage Change

r_t = (p_t / (p_t-1) ) - 1

This defines return at time t as equal to the price at the time t divided by the price at time t-1 minus 1. This is helpful to analyzing the volatility of the stock. 


```python
facebook["returns"] = facebook["Close"].pct_change(1)
```


```python
facebook["returns"].hist(bins=100);
```


![png](output_23_0.png)


### Cumulative Daily Returns
We can see which stock was the most wide ranging in daily returns.

With daily cumulative returns, the question we are trying to answer is the following, if I invested 1 usd in the company at the beginning of time series, how much would is be worth today? This is different than just the stock price at the current day, because it will take into account the daily returns.

df[daily_cumulative_return] = (1 + df[pct_daily_return]).cumprod()


```python
facebook["Cumulative Return"] = (1+facebook["returns"]).cumprod()
```


```python
facebook.index = pd.to_datetime(facebook.index)
```


```python
plt.figure(figsize=(16,8))
plt.title("Cumulative Return")
plt.plot(facebook["Cumulative Return"])
```




    [<matplotlib.lines.Line2D at 0x1a1d8d6940>]




![png](output_27_1.png)



```python
facebook.corr()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Total Traded</th>
      <th>MA30</th>
      <th>MA100</th>
      <th>returns</th>
      <th>Cumulative Return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Open</th>
      <td>1.000000</td>
      <td>0.999797</td>
      <td>0.999724</td>
      <td>0.999535</td>
      <td>0.999535</td>
      <td>-0.449955</td>
      <td>0.283578</td>
      <td>0.994310</td>
      <td>0.981438</td>
      <td>-0.015109</td>
      <td>0.999536</td>
    </tr>
    <tr>
      <th>High</th>
      <td>0.999797</td>
      <td>1.000000</td>
      <td>0.999682</td>
      <td>0.999766</td>
      <td>0.999766</td>
      <td>-0.445842</td>
      <td>0.289132</td>
      <td>0.994601</td>
      <td>0.982242</td>
      <td>-0.007492</td>
      <td>0.999769</td>
    </tr>
    <tr>
      <th>Low</th>
      <td>0.999724</td>
      <td>0.999682</td>
      <td>1.000000</td>
      <td>0.999795</td>
      <td>0.999795</td>
      <td>-0.455724</td>
      <td>0.273871</td>
      <td>0.993672</td>
      <td>0.980424</td>
      <td>-0.005044</td>
      <td>0.999795</td>
    </tr>
    <tr>
      <th>Close</th>
      <td>0.999535</td>
      <td>0.999766</td>
      <td>0.999795</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-0.451143</td>
      <td>0.280846</td>
      <td>0.994009</td>
      <td>0.981126</td>
      <td>0.002697</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Adj Close</th>
      <td>0.999535</td>
      <td>0.999766</td>
      <td>0.999795</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-0.451143</td>
      <td>0.280846</td>
      <td>0.994009</td>
      <td>0.981126</td>
      <td>0.002697</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>-0.449955</td>
      <td>-0.445842</td>
      <td>-0.455724</td>
      <td>-0.451143</td>
      <td>-0.451143</td>
      <td>1.000000</td>
      <td>0.551611</td>
      <td>-0.481777</td>
      <td>-0.497174</td>
      <td>0.125106</td>
      <td>-0.483198</td>
    </tr>
    <tr>
      <th>Total Traded</th>
      <td>0.283578</td>
      <td>0.289132</td>
      <td>0.273871</td>
      <td>0.280846</td>
      <td>0.280846</td>
      <td>0.551611</td>
      <td>1.000000</td>
      <td>0.303425</td>
      <td>0.265978</td>
      <td>-0.006447</td>
      <td>0.296636</td>
    </tr>
    <tr>
      <th>MA30</th>
      <td>0.994310</td>
      <td>0.994601</td>
      <td>0.993672</td>
      <td>0.994009</td>
      <td>0.994009</td>
      <td>-0.481777</td>
      <td>0.303425</td>
      <td>1.000000</td>
      <td>0.990773</td>
      <td>-0.032579</td>
      <td>0.994009</td>
    </tr>
    <tr>
      <th>MA100</th>
      <td>0.981438</td>
      <td>0.982242</td>
      <td>0.980424</td>
      <td>0.981126</td>
      <td>0.981126</td>
      <td>-0.497174</td>
      <td>0.265978</td>
      <td>0.990773</td>
      <td>1.000000</td>
      <td>-0.051608</td>
      <td>0.981126</td>
    </tr>
    <tr>
      <th>returns</th>
      <td>-0.015109</td>
      <td>-0.007492</td>
      <td>-0.005044</td>
      <td>0.002697</td>
      <td>0.002697</td>
      <td>0.125106</td>
      <td>-0.006447</td>
      <td>-0.032579</td>
      <td>-0.051608</td>
      <td>1.000000</td>
      <td>0.002697</td>
    </tr>
    <tr>
      <th>Cumulative Return</th>
      <td>0.999536</td>
      <td>0.999769</td>
      <td>0.999795</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-0.483198</td>
      <td>0.296636</td>
      <td>0.994009</td>
      <td>0.981126</td>
      <td>0.002697</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



I checked the correlation. This is helpful to choose predictor variables.

### Part 4 Machine Learning

While splitting the data into train and validation set, we cannot use random splitting since that will destroy the time component. So here we have set %5 of  data into test and the rest for training.


```python
print("Number of data points in Facebook dataset: {}".format(len(facebook)))
```

    Number of data points in Facebook dataset: 1837


### Linear Regression


```python
train = facebook[["Adj Close","Volume", "Total Traded", "Low"]].iloc[:1735].dropna()
test = facebook[["Adj Close","Volume", "Total Traded", "Low"]].iloc[1735:].dropna()

x_train = train.drop("Adj Close", axis=1)
y_train = train["Adj Close"]

x_test = test.drop("Adj Close", axis=1)
y_test = test["Adj Close"]

model_linear = LinearRegression()
model_linear.fit(x_train,y_train)

preds_linear = model_linear.predict(x_test)

test['Predictions Linear'] = 0
test['Predictions Linear'] = preds_linear

test.index = pd.to_datetime(test.index)
train.index = pd.to_datetime(train.index)

plt.figure(figsize=(16,8))
plt.plot(train["Adj Close"])
plt.plot(test[['Adj Close', 'Predictions Linear']])
plt.legend(labels=["Train", "Test", "Prediction"]);
```


![png](output_34_0.png)



```python
# The coefficients
print('Coefficients: \n', model_linear.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, preds_linear))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, preds_linear))
```

    Coefficients: 
     [ -4.78153419e-09   2.60035601e-10   1.00442788e+00]
    Mean squared error: 2.21
    Variance score: 0.97



```python
plt.figure(figsize=(16,8))
plt.plot(test[['Adj Close', 'Predictions Linear']])
plt.legend(labels=["Test", "Prediction"]);
```


![png](output_36_0.png)


### Ridge Regression


```python
train = facebook[["Adj Close","Volume", "Total Traded", "Low"]].iloc[:1735].dropna()
test = facebook[["Adj Close","Volume", "Total Traded", "Low"]].iloc[1735:].dropna()

x_train = train.drop("Adj Close", axis=1)
y_train = train["Adj Close"]

x_test = test.drop("Adj Close", axis=1)
y_test = test["Adj Close"]

model_ridge = Ridge()
model_ridge.fit(x_train,y_train)

preds_ridge = model_ridge.predict(x_test)

test['Predictions Ridge'] = 0
test['Predictions Ridge'] = preds_ridge

test.index = pd.to_datetime(test.index)
train.index = pd.to_datetime(train.index)

plt.figure(figsize=(16,8))
plt.plot(train["Adj Close"])
plt.plot(test[['Adj Close', 'Predictions Ridge']])
plt.legend(labels=["Train", "Test", "Prediction"]);
```


![png](output_38_0.png)



```python
# The coefficients
print('Coefficients: \n', model_ridge.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, preds_ridge))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, preds_ridge))

```

    Coefficients: 
     [ -4.78233473e-09   2.60046178e-10   1.00442734e+00]
    Mean squared error: 2.21
    Variance score: 0.97



```python
plt.figure(figsize=(16,8))
plt.plot(test[['Adj Close', 'Predictions Ridge']])
plt.legend(labels=["Test", "Prediction"]);
```


![png](output_40_0.png)


### Lasso Regression


```python
train = facebook[["Adj Close","Volume", "Total Traded", "Low"]].iloc[:1735].dropna()
test = facebook[["Adj Close","Volume", "Total Traded", "Low"]].iloc[1735:].dropna()

x_train = train.drop("Adj Close", axis=1)
y_train = train["Adj Close"]

x_test = test.drop("Adj Close", axis=1)
y_test = test["Adj Close"]

model_lasso = Lasso()
model_lasso.fit(x_train,y_train)

preds_lasso = model.predict(x_test)

test['Predictions Lasso'] = 0
test['Predictions Lasso'] = preds_lasso

test.index = pd.to_datetime(test.index)
train.index = pd.to_datetime(train.index)

plt.figure(figsize=(16,8))
plt.plot(train["Adj Close"])
plt.plot(test[['Adj Close', 'Predictions Lasso']])
plt.legend(labels=["Train", "Test", "Prediction"]);
```


![png](output_42_0.png)



```python
# The coefficients
print('Coefficients: \n', model_lasso.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, preds_lasso))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, preds_lasso))
```

    Coefficients: 
     [ -6.16434634e-09   2.78305363e-10   1.00350448e+00]
    Mean squared error: 2.25
    Variance score: 0.97



```python
plt.figure(figsize=(16,8))
plt.plot(test[['Adj Close', 'Predictions Lasso']])
plt.legend(labels=["Test", "Prediction"]);
```


![png](output_44_0.png)

