import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from datetime import datetime
import time

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')  # Apply a single plot style

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load and preview data
aep = pd.read_csv('Input/AEP_hourly.csv', index_col=[0], parse_dates=[0])
aep.head()

# Plotting the initial data
color_pal = sns.color_palette()
aep.plot(style='.', figsize=(10, 5), ms=1, color=color_pal[0], title='AEP MW')
plt.show()

# Feature engineering
cat_type = CategoricalDtype(categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)

def create_features(df, label=None):
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['date_offset'] = (df.date.dt.month * 100 + df.date.dt.day - 320) % 1300
    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], labels=['Spring', 'Summer', 'Fall', 'Winter'])
    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'weekday', 'season']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features(aep, label='AEP_MW')
features_and_target = pd.concat([X, y], axis=1)

fig, ax = plt.subplots(figsize=(15, 7))
sns.boxplot(data=features_and_target.dropna(), x='weekday', y='AEP_MW', hue='season', ax=ax, linewidth=1)
ax.set_title('Power Use MW by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Energy (MW)')
ax.legend(bbox_to_anchor=(1, 1))
plt.show()

# Train-test split
split_date = '2015-01-01'
aep_train = aep.loc[aep.index <= split_date].copy()
aep_test = aep.loc[aep.index > split_date].copy()

# Plot train and test
aep_test.rename(columns={'AEP_MW': 'TEST SET'}).join(aep_train.rename(columns={'AEP_MW': 'TRAINING SET'}), how='outer').plot(figsize=(15, 7), title='AEP MW', style='.', ms=1)
plt.show()

# Prepare data for Prophet model
aep_train_prophet = aep_train.reset_index().rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'})
model = Prophet()
start_time = time.time()
model.fit(aep_train_prophet)
print(f"Model fitting time: {time.time() - start_time:.2f} seconds")

# Forecast on test data
aep_test_prophet = aep_test.reset_index().rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'})
aep_test_fcst = model.predict(aep_test_prophet)

fig, ax = plt.subplots(figsize=(10, 5))
model.plot(aep_test_fcst, ax=ax)
ax.set_title('Prophet Forecast')
plt.show()

model.plot_components(aep_test_fcst)
plt.show()

# Validation with January 2015 Forecast vs Actuals
fig, ax = plt.subplots(figsize=(15, 5))
ax.scatter(aep_test.index, aep_test['AEP_MW'], color='r')
model.plot(aep_test_fcst, ax=ax)
ax.set_xbound(lower=datetime(2015, 1, 1), upper=datetime(2015, 2, 1))
ax.set_ylim(0, 60000)
plt.suptitle('January 2015 Forecast vs Actuals')
plt.show()

# Set the x-axis bounds with datetime objects for the first week of January forecast vs actuals
fig, ax = plt.subplots(figsize=(15, 5))
ax.scatter(aep_test.index, aep_test['AEP_MW'], color='r')
model.plot(aep_test_fcst, ax=ax)
ax.set_xbound(lower=datetime(2015, 1, 1), upper=datetime(2015, 1, 8))
ax.set_ylim(0, 60000)
ax.set_title('First Week of January Forecast vs Actuals')
plt.show()

# Evaluation with Error Metrics
rmse = np.sqrt(mean_squared_error(y_true=aep_test['AEP_MW'], y_pred=aep_test_fcst['yhat']))
mae = mean_absolute_error(y_true=aep_test['AEP_MW'], y_pred=aep_test_fcst['yhat'])
mape = mean_absolute_percentage_error(y_true=aep_test['AEP_MW'], y_pred=aep_test_fcst['yhat'])

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)

# Adding Holidays - Using USFederalHolidayCalendar for holiday data
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

cal = calendar()
holidays = cal.holidays(start=aep.index.min(), end=aep.index.max(), return_name=True)
holiday_df = pd.DataFrame(data=holidays, columns=['holiday']).reset_index().rename(columns={'index': 'ds'})

# Prophet model with holidays
model_with_holidays = Prophet(holidays=holiday_df)
model_with_holidays.fit(aep_train_prophet)

# Predict on test set with holiday model
aep_test_fcst_with_hols = model_with_holidays.predict(aep_test_prophet)

# Plot forecast components with holidays
fig = model_with_holidays.plot_components(aep_test_fcst_with_hols)
plt.show()

# Visualize holiday predictions against actuals for July 4
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(aep_test.index, aep_test['AEP_MW'], color='r')
model_with_holidays.plot(aep_test_fcst_with_hols, ax=ax)
ax.set_xbound(lower=datetime(2015, 7, 1), upper=datetime(2015, 7, 7))
ax.set_ylim(0, 60000)
plt.suptitle('July 4 Predictions vs Actual')
plt.show()

# Error metrics with holidays
rmse_with_holidays = np.sqrt(mean_squared_error(y_true=aep_test['AEP_MW'], y_pred=aep_test_fcst_with_hols['yhat']))
mae_with_holidays = mean_absolute_error(y_true=aep_test['AEP_MW'], y_pred=aep_test_fcst_with_hols['yhat'])
mape_with_holidays = mean_absolute_percentage_error(y_true=aep_test['AEP_MW'], y_pred=aep_test_fcst_with_hols['yhat'])

print("RMSE with Holidays:", rmse_with_holidays)
print("MAE with Holidays:", mae_with_holidays)
print("MAPE with Holidays:", mape_with_holidays)

# Predicting into the future
future = model_with_holidays.make_future_dataframe(periods=365*24, freq='H', include_history=False)
forecast = model_with_holidays.predict(future)

# Plotting future predictions
fig, ax = plt.subplots(figsize=(15, 7))
model_with_holidays.plot(forecast, ax=ax)
ax.set_title('Future Forecast')
plt.show()
