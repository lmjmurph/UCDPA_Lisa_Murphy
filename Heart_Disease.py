## UCDPA PROJECT - LISA MURPHY

## Step 1 - Importing relevant Python libraries for analysis

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from scipy.stats import boxcox


#### Step 2 - Import dataset: import heart disease csv into Pandas dataframe and review dataset

heart_disease = pd.read_csv("heart.csv")
print('DISPLAY FIRST FIVE RECORDS: \n', heart_disease.head(), '')

# Get overview of the structure of the data using the shape attribute:
print('DATA SHAPE: \n', heart_disease.shape, '')

# Get overview of the field types using info()
print('\n\nSUMMARY OF FIELDS:\n', heart_disease.info())

# Create function to describe all fields of dataframe
def describe_all(df):
    print('\nDESCRIPTION OF DATA')
    for i in df:
        print('\n', i, '\n', df[i].describe())
        if df[i].dtype=='object':
            print("Categories:\n", df[i].value_counts(), '\n')

# call describe_all function to describe pre-cleaned dataset
describe_all(heart_disease)


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
describe_all(heart_disease)

# Check for any Missing Values (NaN)
print('\nIDENTIFY MISSING VALUES: \n', heart_disease.isnull().sum(), '')

# Remove duplicates and re-chack shape
heart_disease.drop_duplicates(inplace=True)
print('\nDATA SHAPE AFTER REMOVING DUPLICATES: \n', heart_disease.shape)

# Boxplot of Cholesterol by Sex; hue to differentiate heart disease yes / no
#sns.boxplot(x="Sex", hue="HeartDisease", y="Cholesterol", data=heart_disease)
# adding a title to the plot:
#plt.title('Cholesterol by Sex and Heart Disease', fontsize=14)
# displaying the plot:
#plt.show()

# make a list of any columns containing NaNs
print('\nLIST OF COLUMNS CONTAINING NAN VALUES: ',heart_disease.columns[heart_disease.isna().any()].tolist())


#sns.boxplot(x="Sex", hue="HeartDisease", y="Cholesterol", data=heart_disease)
# adding a title to the plot:
#plt.title('Cholesterol and Heart Disease by Sex', fontsize=14)
# displaying the plot:
#plt.show()


# to facilitate analysis and charts for categorical data vs numerical data divide the dataframe
cat_data = heart_disease.select_dtypes(include='object')
num_data = heart_disease.select_dtypes(exclude='object')
print("\nCategorical fields: \n", cat_data.columns, "\n\nNumerical fields: \n", num_data.columns)

# Look for outliers in numerical fields
def find_outlier(col):
    q1 = heart_disease[col].quantile(0.25)
    q3 = heart_disease[col].quantile(0.75)
    iqr = q3 - q1
    min_val = q1 - 1.5*iqr
    max_val = q3 + 1.5*iqr
    outlier = pd.DataFrame(heart_disease[(heart_disease[col] <= min_val) | (heart_disease[col] >= max_val)])
    print('\nOutliers in ',col, ':\n', outlier.value_counts(outlier['HeartDisease']))
    sns.boxplot(x="Sex", y=col, data=heart_disease)
    # adding a title to the plot:
    plt.title(col, fontsize=14)
    # displaying the plot:
    plt.show()

for i in num_data:
    find_outlier(i)

#  Replace RestingBP zero values with mean
heart_disease['RestingBB'] = heart_disease['RestingBP'].replace (0, heart_disease['RestingBP'].mean(), inplace=True)
print(heart_disease.describe())
find_outlier('RestingBP')


# Cholesterol value = 0 is essentially a missing value - replace with NaN using Numpy
heart_disease['Cholesterol'].replace(0, heart_disease['Cholesterol'].mean(), inplace=True)
find_outlier('Cholesterol')

# find out if any columns contain NaNs
print(heart_disease.columns[heart_disease.isna().any()].tolist())

# display detail of all fields using local describe_all function
describe_all(heart_disease)


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
print(num_data_cols)

# UNHASH AFTER
# plot numercial data using pairplot, distinguishing heart disease
sns.pairplot(heart_disease.loc[:,num_data_cols], hue='HeartDisease')
plt.show()
sns.pairplot(heart_disease.loc[:,num_data_cols])
plt.show()

# UNHASH AFTER
# box plots
#for i in num_data:
#    sns.boxplot(x="Sex", hue="HeartDisease", y=i, data=heart_disease)
#    # adding a title to the plot:
#    plt.title(i, fontsize=14)
#    # displaying the plot:
#    plt.show()

# To use scikit learn we need to encode categorical features - use get_dummies()
heart_disease_encoded = pd.get_dummies(heart_disease, drop_first=True)
print(heart_disease_encoded.head())
print(heart_disease_encoded.info())

# Create feature and target arrays
X = heart_disease_encoded.drop('HeartDisease_Yes', axis=1)
y = heart_disease_encoded['HeartDisease_Yes']
print(X)
print(y)
#y = heart_disease_encoded['HeartDisease_Yes'].values()
#print(y)
#_ = pd.plotting.scatter_matrix(X, c = y, marker = 'D')

#____________________________________
# Prepare for Modelling with Scikit
#____________________________________

# Impute NaNs to replace with Median
imp = SimpleImputer(missing_values=np.NaN, strategy='median')
imp.fit(X)
X_imputed = imp.transform(X)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=42, stratify = y)


#_______________________________________
# Create function to run models and tune hyperparameters
#_______________________________________

def fit_model(steps, parameters):

    pipeline = Pipeline(steps)

    # Instantiate the GridSearchCV object: cv
    cv = GridSearchCV(pipeline, param_grid=parameters)

    # Fit to the training set
    cv.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = cv.predict(X_test)

    # Compute and print metrics
    print("Accuracy: {}".format(cv.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    print("Tuned Model Parameters: {}".format(cv.best_params_))


#==============================
# SINGLE VECTOR CLASSIFIER
#==============================

# Instantiate the Single Vector classifier
svc = SVC()

# Setup the pipeline with a tuple assigned to steps
steps_svc = [('imputation', imp),('scaler',StandardScaler()),
         ('SVM', svc)]

# Create the pipeline
pipeline_svc = Pipeline(steps_svc)

# Fit the pipeline to the train set
pipeline_svc.fit(X_train, y_train)

# Predict the labels of the test set
y_pred_svc = pipeline_svc.predict(X_test)

# print the confusion matrix
print(confusion_matrix(y_test, y_pred_svc))

# Print classification report for SVC model
print(classification_report(y_test,y_pred_svc))


# Instantiate Logistic Regression
logreg = LogisticRegression()


# Setup the pipeline with a tuple assigned to steps
steps_logreg = [('imputation', imp),('scaler',StandardScaler()),
         ('LogReg', logreg)]

# Create the pipeline
pipeline_logreg = Pipeline(steps_logreg)

# Create training and test sets
pipeline_logreg.fit(X_train, y_train)

# Print score for applying Logistic Regression model to test data
print(pipeline_logreg.score(X_test,y_test))

# Predict the labels of the test set
y_pred_logreg = pipeline_logreg.predict(X_test)
print(y_pred_logreg)

# print the confusion matrix
print(confusion_matrix(y_test, y_pred_logreg))

# Compute metrics
print("CLASSIFICATION REPORT FOR LOGISTIC REGRESSION MODEL:\n", classification_report(y_test,y_pred_logreg))

y_pred_logreg_prob = pipeline_logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_logreg_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show();

print("LOGISTIC REGRESSION AUC SCORE: ", roc_auc_score(y_test, y_pred_logreg_prob))


# K NEIGHBORS CLASSIFIER MODEL

steps2 = [('imputation', imp),('scaler',StandardScaler()),('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps2)
parameters = {"knn__n_neighbors": range(1, 50)}
cv = GridSearchCV(pipeline, param_grid=parameters)
print(cv)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

print(cv.best_params_)
print(cv.score(X_test, y_test))
print(classification_report(y_test, y_pred))


