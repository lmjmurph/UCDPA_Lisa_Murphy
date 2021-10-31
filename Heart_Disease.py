## UCDPA PROJECT - LISA MURPHY

## Step 1 - Importing relevant Python libraries for analysis

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import cross_val_score
from scipy.stats import boxcox

#===============IMPORT DATASET=========================

# import heart disease csv into Pandas dataframe

heart_disease = pd.read_csv("heart.csv")
pd.options.display.width= None
pd.options.display.max_columns= None

print(f"DISPLAY FIRST FIVE RECORDS: \n {heart_disease.head()}")


#===============OVERVIEW OF DATA=======================

# Get overview of the structure of the data using the shape attribute:
print(f"DATA SHAPE:\n{heart_disease.shape}")

# Get overview of the field types using describe()
print(f"\n\nSUMMARY OF FIELDS:\n{heart_disease.describe(include='all')}")


#### Step 3 - Cleaning and Formatting the data

# Convert Fasting Blood Sugar from numerical to categorical (1 = Male, 0 = Female)
heart_disease['FastingBS'] = heart_disease['FastingBS'].astype('object')
assert heart_disease['FastingBS'].dtype == 'object'
heart_disease['FastingBS'] = heart_disease['FastingBS'].replace(1,'Male')
heart_disease['FastingBS'] = heart_disease['FastingBS'].replace(0,'Female')

# Rename the FastingBS column for clarity
heart_disease.rename(columns = {'FastingBS':'Fasting BS >120'},inplace=True)

# Convert Heart Disease from numerical to categorical (1 = Yes, 0 = No)
heart_disease['HeartDisease'] = heart_disease['HeartDisease'].astype('object')
assert heart_disease['HeartDisease'].dtype == 'object'
heart_disease['HeartDisease'] = heart_disease['HeartDisease'].replace(1,'Yes')
heart_disease['HeartDisease'] = heart_disease['HeartDisease'].replace(0,'No')

# review after changes
print (f"\n\nSUMMARY OF FIELDS AFTER CHANGES MADE: \n{heart_disease.describe(include='all')}")


# Check for any Missing Values (NaN)
print(f"\nIDENTIFY MISSING VALUES: \n{heart_disease.isnull().sum()}")

# Remove duplicates and re-chack shape
heart_disease.drop_duplicates(inplace=True)
print(f"\nDATA SHAPE AFTER REMOVING DUPLICATES:\n{heart_disease.shape}")

# Boxplot of Cholesterol by Sex; hue to differentiate heart disease yes / no
#sns.boxplot(x="Sex", hue="HeartDisease", y="Cholesterol", data=heart_disease)
# adding a title to the plot:
#plt.title('Cholesterol by Sex and Heart Disease', fontsize=14)
# displaying the plot:
#plt.show()

# make a list of any columns containing NaNs
print(f"\nLIST OF COLUMNS CONTAINING NAN VALUES: {heart_disease.columns[heart_disease.isna().any()].tolist()}")


#sns.boxplot(x="Sex", hue="HeartDisease", y="Cholesterol", data=heart_disease)
# adding a title to the plot:
#plt.title('Cholesterol and Heart Disease by Sex', fontsize=14)
# displaying the plot:
#plt.show()


# to facilitate analysis and charts for categorical data vs numerical data divide the dataframe
cat_data = heart_disease.select_dtypes(include='object')
num_data = heart_disease.select_dtypes(exclude='object')
print(f"\nCategorical fields: \n{cat_data.columns}\n\nNumerical fields: \n{num_data.columns}")

# Look for inconsistent values in categorical fields
for i in cat_data.columns:
    print(f"\nCategories in {i}:\n{heart_disease[i].value_counts()}")

# Look for outliers in numerical fields
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

for i in num_data:
    find_outlier(i)

#  Replace RestingBP zero values with mean
heart_disease['RestingBP'].replace(0, heart_disease['RestingBP'].mean(), inplace=True)
print(heart_disease.describe(include='all'))
find_outlier('RestingBP')


# Cholesterol value = 0 is essentially a missing value - replace with mean
heart_disease['Cholesterol'].replace(0, heart_disease['Cholesterol'].mean(), inplace=True)
find_outlier('Cholesterol')

# find out if any columns contain NaNs
print(f"\nColumns containing NaNs: {heart_disease.columns[heart_disease.isna().any()].tolist()}")

# display detail of all fields using local describe_all function
#describe_all(heart_disease)

f, axes = plt.subplots(3, 2, figsize=(20,15))
f.patch.set_facecolor('#E5FAFA')

for i in enumerate(num_data.columns):
    print (i)
    plt.subplot(3,2, i[0]+1)
    sns.histplot(x=i[1],data=heart_disease, kde=True)
    plt.xlabel('')
    plt.ylabel('# of records')
    plt.yticks(fontsize=10, color='black')
    plt.xticks(fontsize=10, color='black')
    plt.box(False)
    plt.title(i[1], fontsize=18, color='black')
    plt.tight_layout(pad=5.0)



# create function to do countplot of different features vs Heart Disease
def cplot(col):
    plt.figure()
    sns.countplot(x=col, hue='HeartDisease', data=heart_disease, palette='RdBu')
    plt.xticks(list(range(heart_disease[col].unique().size)), list(heart_disease[col].unique()))
    plt.show()

#### UNHASH AFTER
#for i in cat_data:
#    cplot(i)

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

# box plots
#for i in num_data:
#    sns.boxplot(x="Sex", hue="HeartDisease", y=i, data=heart_disease)
#    # adding a title to the plot:
#    plt.title(i, fontsize=14)
#    # displaying the plot:
#    plt.show()


### PREPARE FOR MACHINE LEARNING WITH SCIKIT.LEARN



# To use scikit learn we need to encode categorical features - use get_dummies()
heart_disease_encoded = pd.get_dummies(heart_disease, drop_first=True)
print(heart_disease_encoded.head())
print('Encoded info: ',heart_disease_encoded.info())

# Create feature and target arrays
X = heart_disease_encoded.drop('HeartDisease_Yes', axis=1)
y = heart_disease_encoded['HeartDisease_Yes']

#y = heart_disease_encoded['HeartDisease_Yes'].values()
#print(y)
#_ = pd.plotting.scatter_matrix(X, c = y, marker = 'D')

#____________________________________
# Prepare for Modelling with Scikit
#____________________________________


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42, stratify = y)

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#_______________________________________
# Create function to run models and tune hyperparameters
#_______________________________________

def fit_model(estimator, parameters):
    cv = GridSearchCV(estimator, param_grid=parameters)
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

