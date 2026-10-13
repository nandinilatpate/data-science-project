# Metro Pulse: Predicting and Visualizing Urban Metro Ridership Patterns

# Bengaluru Namma Metro Ridership Dataset




# ======================================================================
# 🚇 Metro Pulse: Predicting and Visualizing Urban Metro Ridership Patterns

#**Dataset:** Bengaluru Namma Metro Ridership Dataset  
#**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
#**Project level:** Student mini-project / Data Science practical

#Objective
#Analyze daily Namma Metro ridership patterns, visualize trends and distributions, engineer time-based features, predict total daily ridership, and evaluate multiple regression models.
# ======================================================================


# ======================================================================
# 1. Import Libraries

#We use Pandas for data handling, NumPy for numerical operations, Matplotlib and Seaborn for visualization, and Scikit-learn for machine learning.
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid')


# ======================================================================
# 2. Load the Dataset
# ======================================================================

file_path = 'NammaMetro_Ridership_Dataset.csv'
df = pd.read_csv("NammaMetro_Ridership_Dataset.csv")

print('Shape:', df.shape)
print(df.head())


# ======================================================================
# 3. Understand the Dataset
#Check column names, data types, summary statistics, missing values and duplicate rows.
# ======================================================================

print('Columns:')
print(df.columns.tolist())

print('\nData types:')
print(df.dtypes)

print('\nSummary statistics:')
print(df.describe())

print('\nMissing values:')
print(df.isnull().sum())

print('\nDuplicate rows:', df.duplicated().sum())


# ======================================================================
# 4. Data Cleaning and Date Conversion
#The date is converted to a proper datetime format. The dataset is then sorted chronologically.
# ======================================================================

df['Record Date'] = pd.to_datetime(df['Record Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Record Date']).copy()
df = df.sort_values('Record Date').reset_index(drop=True)

print('Date range:', df['Record Date'].min().date(), 'to', df['Record Date'].max().date())


# ======================================================================
# 5. Create Total Ridership

#For this project, total daily ridership is defined using the five main ticketing categories available in the dataset:

#**Total Ridership = Total Smart Cards + Total Tokens + Total NCMC + Group Ticket + Total QR**

#The detailed QR source columns are kept for analysis, but they are not added again because they are components of Total QR.
# ======================================================================

ridership_components = [
    'Total Smart Cards', 'Total Tokens', 'Total NCMC',
    'Group Ticket', 'Total QR'
]

df['Total Ridership'] = df[ridership_components].sum(axis=1)

print('Average daily ridership:', round(df['Total Ridership'].mean(), 2))
print('Maximum daily ridership:', df['Total Ridership'].max())
print('Minimum daily ridership:', df['Total Ridership'].min())


# ======================================================================
# 6. Feature Engineering
# ======================================================================

df['Year'] = df['Record Date'].dt.year
df['Month'] = df['Record Date'].dt.month
df['Day'] = df['Record Date'].dt.day
df['Day_of_Week'] = df['Record Date'].dt.dayofweek
df['Day_Name'] = df['Record Date'].dt.day_name()
df['Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)

print(df[['Record Date', 'Year', 'Month', 'Day', 'Day_Name', 'Weekend', 'Total Ridership']].head())


# ======================================================================
# 7. Basic Statistical Analysis
# ======================================================================

stats = df['Total Ridership'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(stats)

print('\nHighest-ridership day:')
print(df.loc[df['Total Ridership'].idxmax(), ['Record Date', 'Total Ridership']])

print('\nLowest-ridership day:')
print(df.loc[df['Total Ridership'].idxmin(), ['Record Date', 'Total Ridership']])


# ======================================================================
# 8. Visualization 1 — Daily Ridership Trend (Matplotlib)
# ======================================================================

plt.figure(figsize=(14, 5))
plt.plot(df['Record Date'], df['Total Ridership'], linewidth=1.5)
plt.title('Daily Namma Metro Ridership Trend')
plt.xlabel('Date')
plt.ylabel('Total Ridership')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ======================================================================
# 9. Visualization 2 — Monthly Average Ridership
# ======================================================================

monthly = df.groupby('Month', as_index=False)['Total Ridership'].mean()

plt.figure(figsize=(10, 5))
plt.bar(monthly['Month'], monthly['Total Ridership'])
plt.title('Average Ridership by Month')
plt.xlabel('Month')
plt.ylabel('Average Ridership')
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()


# ======================================================================
# 10. Visualization 3 — Year-wise Ridership
# ======================================================================

yearly = df.groupby('Year', as_index=False)['Total Ridership'].mean()

plt.figure(figsize=(8, 5))
plt.bar(yearly['Year'].astype(str), yearly['Total Ridership'])
plt.title('Average Daily Ridership by Year')
plt.xlabel('Year')
plt.ylabel('Average Daily Ridership')
plt.tight_layout()
plt.show()


# ======================================================================
# 11. Visualization 4 — Weekday vs Weekend
# ======================================================================

weekend_avg = df.groupby('Weekend')['Total Ridership'].mean()
labels = ['Weekday', 'Weekend']

plt.figure(figsize=(7, 5))
plt.bar(labels, [weekend_avg.get(0, 0), weekend_avg.get(1, 0)])
plt.title('Average Ridership: Weekday vs Weekend')
plt.xlabel('Day Type')
plt.ylabel('Average Ridership')
plt.tight_layout()
plt.show()

print(pd.DataFrame({
    'Day Type': labels,
    'Average Ridership': [weekend_avg.get(0, 0), weekend_avg.get(1, 0)]
}))


# ======================================================================
# 12. Visualization 5 — Ridership Distribution (Seaborn Histogram)
# ======================================================================

plt.figure(figsize=(9, 5))
sns.histplot(df['Total Ridership'], kde=True)
plt.title('Distribution of Daily Metro Ridership')
plt.xlabel('Total Ridership')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ======================================================================
# 13. Visualization 6 — Box Plot for Outliers
# ======================================================================

plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Total Ridership'])
plt.title('Box Plot of Daily Ridership')
plt.xlabel('Total Ridership')
plt.tight_layout()
plt.show()


# ======================================================================
# 14. Visualization 7 — Ridership by Day of Week
# ======================================================================

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_avg = df.groupby('Day_Name')['Total Ridership'].mean().reindex(day_order)

plt.figure(figsize=(10, 5))
sns.barplot(x=day_avg.index, y=day_avg.values)
plt.title('Average Ridership by Day of Week')
plt.xlabel('Day')
plt.ylabel('Average Ridership')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ======================================================================
# 15. Visualization 8 — Correlation Heatmap
# ======================================================================

corr_cols = [
    'Total Smart Cards', 'Total Tokens', 'Total NCMC',
    'Group Ticket', 'Total QR', 'Total Ridership'
]
corr = df[corr_cols].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Ridership Variables')
plt.tight_layout()
plt.show()


# ======================================================================
# 16. Machine Learning — Ridership Prediction

#To avoid data leakage, we do **not** use ticket-category totals as input features because they directly contribute to the calculated target `Total Ridership`.

#Instead, we predict ridership from calendar/time features.
# ======================================================================

features = ['Year', 'Month', 'Day', 'Day_of_Week', 'Weekend']
target = 'Total Ridership'

X = df[features]
y = df[target]

# Chronological split is more appropriate for time-based data.
split_index = int(len(df) * 0.80)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print('Training records:', len(X_train))
print('Testing records:', len(X_test))


# ======================================================================
# 17. Train Three Regression Models
# ======================================================================

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, pred),
        'MSE': mean_squared_error(y_test, pred),
        'R2 Score': r2_score(y_test, pred)
    })

results_df = pd.DataFrame(results).sort_values('R2 Score', ascending=False)
print(results_df)


# ======================================================================
# 18. Actual vs Predicted Ridership
# ======================================================================

best_model_name = results_df.iloc[0]['Model']
best_pred = predictions[best_model_name]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_pred, alpha=0.7)
plt.xlabel('Actual Ridership')
plt.ylabel('Predicted Ridership')
plt.title(f'Actual vs Predicted Ridership — {best_model_name}')
plt.tight_layout()
plt.show()

print('Best model:', best_model_name)


# ======================================================================
# 19. Predict Ridership for a New Date

#Change the date below to test the trained best model on a new calendar date.
# ======================================================================

new_date = pd.Timestamp('2026-09-01')

new_features = pd.DataFrame({
    'Year': [new_date.year],
    'Month': [new_date.month],
    'Day': [new_date.day],
    'Day_of_Week': [new_date.dayofweek],
    'Weekend': [1 if new_date.dayofweek >= 5 else 0]
})

best_model = models[best_model_name]
new_prediction = best_model.predict(new_features)[0]

print('Date:', new_date.date())
print('Predicted ridership:', round(new_prediction))


# ======================================================================
# 20. Key Findings Template

#After running the notebook, record the exact numerical findings from your output. Suggested findings to investigate:

#1. The dataset contains 460 daily records covering the available period.
#2. Check the overall mean, median and standard deviation of daily ridership.
#3. Identify the date with the highest ridership.
#4. Identify the date with the lowest ridership.
#5. Compare average weekday and weekend ridership.
#6. Identify the month with the highest average ridership.
#7. Identify the month with the lowest average ridership.
#8. Use the box plot to discuss possible outliers.
#9. Use the heatmap to identify strong relationships among ridership variables.
#10. Compare MAE, MSE and R² to select the best prediction model.
# ======================================================================


# ======================================================================
# 21. Conclusion

#Metro Pulse demonstrates a complete data-science workflow on Bengaluru Namma Metro ridership data. The project cleans and transforms daily records, performs statistical analysis and EDA, visualizes ridership patterns using Matplotlib and Seaborn, and applies regression models from Scikit-learn to predict daily ridership. The model comparison using MAE, MSE and R² helps identify the most suitable model among the tested approaches.

#Future Scope
#- Add holidays and public-event information.
#- Add weather and rainfall data.
#- Add station-wise and line-wise ridership.
#- Use real-time passenger counts for live crowd prediction.
#- Develop a dashboard for metro authorities and passengers.
# ======================================================================
