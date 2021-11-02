## UCDPA PROJECT - LISA MURPHY

## Step 1 - Importing relevant Python libraries for analysis

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


#===============IMPORT DATASET=========================

# Set display options to allow all columns to be viewed
pd.options.display.width= None
pd.options.display.max_columns= None

# import heart disease csv into Pandas dataframe
heart_disease = pd.read_csv("heart.csv")


#===============OVERVIEW OF DATA=======================

# Get overview of the structure of the data using the shape attribute:

print(f"DATA SHAPE:\n{heart_disease.shape}")

print(f"DISPLAY FIRST FIVE RECORDS: \n {heart_disease.head()}")

# Get overview of the field types using describe()
print(f"\n\nSUMMARY OF DATA:\n{heart_disease.describe(include='all')}")


#### CLEANING AND FORMATTING THE DATA

# Convert Fasting Blood Sugar from numerical to categorical (1 = Male, 0 = Female)
heart_disease['FastingBS'] = heart_disease['FastingBS'].astype('object')
assert heart_disease['FastingBS'].dtype == 'object'
heart_disease['FastingBS'] = heart_disease['FastingBS'].replace(1,'Yes')
heart_disease['FastingBS'] = heart_disease['FastingBS'].replace(0,'No')

# Rename the FastingBS column for clarity
heart_disease.rename(columns = {'FastingBS':'Fasting BS >120'},inplace=True)

# Convert Heart Disease from numerical to categorical (1 = Yes, 0 = No)
heart_disease['HeartDisease'] = heart_disease['HeartDisease'].astype('object')
assert heart_disease['HeartDisease'].dtype == 'object'
heart_disease['HeartDisease'] = heart_disease['HeartDisease'].replace(1,'Yes')
heart_disease['HeartDisease'] = heart_disease['HeartDisease'].replace(0,'No')

# review after changes
print (f"\n\nSUMMARY OF FIELDS AFTER CHANGES MADE: \n{heart_disease.describe(include='all')}")

# Remove duplicates and re-chack shape
heart_disease.drop_duplicates(inplace=True)
print(f"\nDATA SHAPE AFTER REMOVING DUPLICATES:\n{heart_disease.shape}")

# Check for any Missing Values (NaN)
print(f"\nIDENTIFY MISSING VALUES: \n{heart_disease.isnull().sum()}")

# Replace 0 values in RestingBP and Cholesterol with NaNs
heart_disease['RestingBP'].replace([0],np.nan, inplace=True)
heart_disease['Cholesterol'].replace([0],np.nan, inplace=True)
print(f"\nRECHECK MISSING VALUES: \n{heart_disease.isnull().sum()}")

# make a list of any columns containing NaNs
print(f"\nLIST OF COLUMNS CONTAINING NAN VALUES: {heart_disease.columns[heart_disease.isna().any()].tolist()}")

# Replace RestingBP NaN values with mean (histogram shows that median value could be misleading)
heart_disease['RestingBP'] = heart_disease['RestingBP'].fillna(heart_disease['RestingBP'].mean())

# Replace Cholesterol NaN values with median of cholesterol after segmenting by HeartDisease(less sensitive to outliers)
# This was the least worst option but creates an inconsistent spike in the distribution - future iteration should attempt to segment further?
heart_disease_yes = heart_disease.loc[heart_disease['HeartDisease'] == 'Yes']
print("Shape of Yes ",heart_disease_yes.shape)
heart_disease_no = heart_disease.loc[heart_disease['HeartDisease'] == 'No']
print("Shape of No ",heart_disease_no.shape)

heart_disease_yes['Cholesterol'].fillna(heart_disease_yes['Cholesterol'].median(), inplace=True)
heart_disease_no['Cholesterol'].fillna(heart_disease_no['Cholesterol'].median(), inplace = True)
heart_disease = pd.DataFrame.merge(heart_disease_yes, heart_disease_no, how="outer")
print("Shape of all ",heart_disease.shape)

print(heart_disease.describe(include='all'))

# to facilitate analysis and charts for categorical data vs numerical data divide the dataframe
cat_data = heart_disease.select_dtypes(include='object')
num_data = heart_disease.select_dtypes(exclude='object')
print(f"\nCategorical fields: \n{cat_data.columns}\n\nNumerical fields: \n{num_data.columns}")

# Look for inconsistent values in categorical fields
for i in cat_data.columns:
    print(f"\nCategories in {i}:\n{heart_disease[i].value_counts()}")


# Look for outliers in numerical fields and visualise using boxplots
def find_outlier(col):
    q1 = heart_disease[col].quantile(0.25)
    q3 = heart_disease[col].quantile(0.75)
    iqr = q3 - q1
    min_val = q1 - 1.5*iqr
    max_val = q3 + 1.5*iqr
    outlier = pd.DataFrame(heart_disease[(heart_disease[col] <= min_val) | (heart_disease[col] >= max_val)])
    print(f"\nNumber of outliers in {col}:\n {outlier.value_counts(outlier['HeartDisease'])}")
    sns.boxplot(x="Sex", y=col, data=heart_disease)
    # adding a title to the plot:
    plt.title(col, fontsize=14)
    # displaying the plot:
    plt.show()

# Figure 1 and detail - iterate through numeric columns to identify outliers
for i in num_data:
    find_outlier(i)

for i in heart_disease.columns:
    ct = pd.crosstab(heart_disease[i], heart_disease["HeartDisease"])
    plt.figure(figsize = (16, 16))
    plt.title(f"Crosstab of {i} vs heart disease", fontsize = 20)
    sns.heatmap(ct, cmap = "BuPu", cbar = True, fmt = "g", cbar_kws={'label': 'Number of occurences', 'orientation': 'horizontal'})

f, axes = plt.subplots(3, 2, figsize=(15, 10))
f.patch.set_facecolor('#E5FAFA')
plt.xticks([]),plt.yticks([])
plt.box(False)

for i in enumerate(num_data.columns):
    plt.subplot(3,2, i[0]+1)
    sns.histplot(x=i[1],data=heart_disease, kde=True)
    plt.xlabel('')
    plt.ylabel('# of records')
    plt.yticks(fontsize=10, color='black')
    plt.xticks(fontsize=10, color='black')
    plt.box(False)
    plt.title(i[1], fontsize=18, color='black')
    plt.tight_layout(pad=5.0)


# find out if any columns contain NaNs
print(f"\nColumns containing NaNs: {heart_disease.columns[heart_disease.isna().any()].tolist()}")

# Heat map of numeric data
plt.figure(figsize=(16,9))
ax = sns.heatmap(num_data.corr(),annot = True,cmap = 'YlGnBu')
plt.show()

# create function to do countplot of different features to see how well each category is represented
def cplot(col):
    plt.figure()
    sns.countplot(x=col, data=heart_disease)
    plt.xticks(list(range(heart_disease[col].unique().size)), list(heart_disease[col].unique()))
    plt.show()

for i in cat_data:
    cplot(i)

# create function to do countplot of different features vs Heart Disease
def cplot(col):
    plt.figure()
    sns.countplot(x=col, hue='HeartDisease', data=heart_disease)
    plt.xticks(list(range(heart_disease[col].unique().size)), list(heart_disease[col].unique()))
    plt.show()

for i in cat_data:
    cplot(i)

# identify the non-categorical columns for plotting
num_data_cols = num_data.columns.values


# add Heart Disease column to use as hue in plots
num_data_cols = np.append(num_data_cols,'HeartDisease')
print(f"\nColumns to use in pair plots\n{num_data_cols}")

# plot numerical data using pairplot, distinguishing heart disease
sns.pairplot(heart_disease.loc[:,num_data_cols], hue='HeartDisease')
plt.show()
sns.pairplot(heart_disease.loc[:,num_data_cols])
plt.show()


### PREPARE FOR MACHINE LEARNING WITH SCIKIT.LEARN

# To use scikit learn we need to encode categorical features - use get_dummies()
heart_disease_encoded = pd.get_dummies(heart_disease, drop_first=True)
print(heart_disease_encoded.head())
print('Encoded info: ',heart_disease_encoded.info())

# Create feature and target arrays
X = heart_disease_encoded.drop('HeartDisease_Yes', axis=1)
y = heart_disease_encoded['HeartDisease_Yes']


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42, stratify = y)

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#_______________________________________
# Create function to run models and tune hyper parameters
#_______________________________________

def fit_model(estimator, parameters):
    cv = GridSearchCV(estimator, param_grid=parameters, cv=5)
    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)
    print(f"\nBest Parameter for {estimator} model: {cv.best_params_}")
    print(f"\nCLASSIFICATION REPORT FOR {estimator} MODEL:\n\n{classification_report(y_test, y_pred)}")

# Fit KNN model
knn = KNeighborsClassifier()
knn_params = {"n_neighbors": range(1, 50)}
fit_model(knn, knn_params)


# Fit Logistic Regression model
lr = LogisticRegression()
lr_params = {"C": np.logspace(-4,4,50)}
fit_model(lr, lr_params)

# Fit SVM model
svc = SVC()
print(svc.get_params().keys())
svc_params = {'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']}
fit_model(svc, svc_params)

