# Final Project of Data Analysis Process

# Central Research Question: How does the number of reviews relate to the price, availability, or other features of the listing?

# 1. Access the Dataset
import pandas as pd
import numpy as np
df = pd.read_csv('AB_NYC_2019.csv')

# 2. Data Cleaning
# 1) Analyze your dataset's structure using descriptive functions.
# Displaying the first few rows of the dataframe
pd.set_option('display.max_columns', None)
print(df.head())
# Dimensions of Dataset
print(df.info())
print(df.shape)
# Data Types of Dataset
print(df.dtypes)
# Descriptive Statistics of Dataset
#print(df.describe())

#2) Identify and handle missing values, possibly using DataFrame's dropna() or fillna() methods.
# Checking for missing values
print(df.isnull().sum())
# Since research question focuses on number_of_reviews, price and availability_365, it is important to handle missing value strategically:

# Drop rows with missing important numerical fields
df = df.dropna(subset=['price', 'number_of_reviews', 'availability_365'])

# Fill reviews_per_month with 0
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Drop last_review
df = df.drop(columns=['last_review'])

# Fill minor text missing values
df['name'] = df['name'].fillna('Unknown')
df['host_name'] = df['host_name'].fillna('Unknown')

# Verify again
print(df.isnull().sum())

# 3. Detect and treat outliers using techniques such as the IQR score.
# Checking for outliers and treating them
#print(df.describe())
''' The summary statistics suggest that most listings have low host listing counts and moderate availability,
 but the presence of very high maximum values indicates potential outliers,
 which can be further analyzed using advanced techniques in later steps.'''
# Detect and treat outliers using IQR
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Identify outliers
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print("Number of outliers:", outliers.shape[0])

'''Outliers were identified using the IQR method on the price variable. 
The calculated lower bound (-90) and upper bound (334) indicate that listings priced above 334 are considered outliers.
A total of 2972 listings fall outside this range, suggesting the presence of high-priced listings that may influence the analysis.'''

# 3. Exploratory Data Analysis (EDA)
# 1) Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# 2)Generate Descriptive Statistics
print(df.describe())

''' The descriptive statistics show that the average price is around the dataset mean (with values ranging from very low to a maximum above 300),
 indicating high variability in listing prices. Similarly, availability ranges from 0 to 365 days, showing that while many listings have limited availability,
 some are available throughout the year.'''

# 3) Investigate relationships using correlation

corr = df[['number_of_reviews', 'price', 'availability_365']].corr()
# Print correlation matrix
print(corr)

# Visualize correlation using heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
'''The correlation results show a very weak negative relationship between number_of_reviews and price, indicating that review count has little impact on pricing. 
There is a weak positive relationship between number_of_reviews and availability_365, 
suggesting that listings with more reviews tend to have slightly higher availability, though the effect is minimal. '''

# 4) Visualize the Data
#  Create scatter plots
# Create subplots (1 row, 3 columns)
plt.figure(figsize=(18,5))

# Plot 1: Reviews vs Price
plt.subplot(1, 3, 1)
sns.scatterplot(x='number_of_reviews', y='price', data=df)
plt.title('Reviews vs Price')

# Plot 2: Reviews vs Availability
plt.subplot(1, 3, 2)
sns.scatterplot(x='number_of_reviews', y='availability_365', data=df)
plt.title('Reviews vs Availability')

# Plot 3: Price vs Availability
plt.subplot(1, 3, 3)
sns.scatterplot(x='price', y='availability_365', data=df)
plt.title('Price vs Availability')

# Adjust layout
plt.tight_layout()

# Show all plots in one image
plt.show()
'''The scatter plots show no strong relationship between the variables, with data points widely scattered.
This indicates weak or minimal association between reviews, price, and availability'''

# Create subplots for histograms
plt.figure(figsize=(18,5))

# Histogram 1: Number of Reviews
plt.subplot(1, 3, 1)
sns.histplot(df['number_of_reviews'], bins=50, kde=True)
plt.title('Distribution of Number of Reviews')

# Histogram 2: Price
plt.subplot(1, 3, 2)
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribution of Price')

# Histogram 3: Availability
plt.subplot(1, 3, 3)
sns.histplot(df['availability_365'], bins=50, kde=True)
plt.title('Distribution of Availability')

# Adjust layout
plt.tight_layout()

# Save image (optional)
plt.savefig('histograms.png')

# Show plot
plt.show()

'''The histograms show that number_of_reviews and price are highly right-skewed, with most values concentrated at lower ranges.
Availability is more evenly distributed but still shows variation across listings.'''

# Create subplots for boxplots
plt.figure(figsize=(18,5))

# Boxplot 1: Number of Reviews
plt.subplot(1, 3, 1)
sns.boxplot(x=df['number_of_reviews'])
plt.title('Number of Reviews')

# Boxplot 2: Price
plt.subplot(1, 3, 2)
sns.boxplot(x=df['price'])
plt.title('Price')

# Boxplot 3: Availability
plt.subplot(1, 3, 3)
sns.boxplot(x=df['availability_365'])
plt.title('Availability')

# Adjust layout
plt.tight_layout()

# Save image (optional)
plt.savefig('boxplots.png')

# Show plot
plt.show()

'''The boxplots show the presence of several outliers, especially in price and number_of_reviews.
Most values are concentrated at lower ranges, with a few extreme values extending the distribution.'''

# Explore number of reviews with respect to 'room_type' and 'neighbourhood_group'.
# Create subplots (1 row, 2 columns)
plt.figure(figsize=(14,5))

# Plot 1: Reviews by Room Type
plt.subplot(1, 2, 1)
sns.boxplot(x='room_type', y='number_of_reviews', data=df)
plt.title('Reviews by Room Type')
plt.xticks(rotation=45)

# Plot 2: Reviews by Neighbourhood Group
plt.subplot(1, 2, 2)
sns.boxplot(x='neighbourhood_group', y='number_of_reviews', data=df)
plt.title('Reviews by Neighbourhood Group')
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save image (optional)
plt.savefig('reviews_category.png')

# Show plot
plt.show()
'''The boxplots show that the number of reviews varies across room types and neighbourhoods, 
with some categories having higher median reviews and greater variability.'''

# 4. Advanced Data Analysis Techniques
# Predictive Model: Predict number_of_reviews using listing features

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Select features and target variable
X = df[['price', 'availability_365', 'room_type', 'neighbourhood_group']]
y = df['number_of_reviews']

# Convert categorical variables into numerical variables using one-hot encoding
X = pd.get_dummies(X, columns=['room_type', 'neighbourhood_group'], drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

'''Insights: The model shows a low R-squared value (0.03), indicating that the selected features explain very little variation in the number of reviews.
The error values (MAE ≈ 26.6 and RMSE ≈ 44) suggest that predictions are not highly accurate.
This indicates that price, availability, room type, and neighbourhood group are not strong predictors of review count.
Other factors may play a more important role in influencing the number of reviews.'''