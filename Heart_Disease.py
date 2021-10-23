## Step 1 - Importing relevant Python libraries for analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import boxcox


## Step 2 - Import dataset: import heart disease csv into Pandas dataframe

heart_disease = pd.read_csv("heart.csv")
print('DISPLAY FIRST FIVE RECORDS: \n', heart_disease.head(), '')

# Get overview of the structure of the data using the shape attribute:
print('DATA SHAPE: \n', heart_disease.shape, '')

# Create a function to get a summary about the data with all the basic summary statistics

def describe_all(df):
    print('\nDESCRIPTION OF DATA')
    for i in df:
        print('\n', i, '\n', df[i].describe())
        if df[i].dtype=='object':
            print("Categories:\n", df[i].value_counts(), '\n')

describe_all(heart_disease)

##------------------------------------------
# Step 3 - Cleaning and Formatting the data
##------------------------------------------

# Convert Fasting Blood Sugar from numerical to categorical (1 = True, 0 = False)
heart_disease['FastingBS'] = heart_disease['FastingBS'].astype('object')
assert heart_disease['FastingBS'].dtype == 'object'


# Convert Heart Disease from numerical to categorical (1 = True, 0 = False)
heart_disease['HeartDisease'] = heart_disease['HeartDisease'].astype('object')
assert heart_disease['HeartDisease'].dtype == 'object'


# Check for any Missing Values (NaN)
print('\nIDENTIFY MISSING VALUES: \n', heart_disease.isnull().sum(), '')

# Remove duplicates and re-chack shape
heart_disease.drop_duplicates(inplace=True)
print('\nDATA SHAPE AFTER REMOVING DUPLICATES: \n', heart_disease.shape, '')


sns.boxplot(x="Sex", hue="HeartDisease", y="Cholesterol", data=heart_disease)
# adding a title to the plot:
plt.title('Renme', fontsize=14)
# displaying the plot:
plt.show()

# Look closer at Cholesterol values - min value = 0 doesn't make sense
print(heart_disease['Cholesterol'].value_counts().sort_index())
heart_disease['Cholesterol'].replace([0],np.nan, inplace=True)

describe_all(heart_disease)

sns.boxplot(x="Sex", hue="HeartDisease", y="Cholesterol", data=heart_disease)
# adding a title to the plot:
plt.title('Renme', fontsize=14)
# displaying the plot:
plt.show()


# to facilitate charts for categorical data vs numerical data divide the dataframe
cat_data = heart_disease.select_dtypes(include='object')
num_data = heart_disease.select_dtypes(exclude='object')
print("\nCategorical fields: \n", cat_data.columns, "\n\nNumerical fields: \n", num_data.columns)

# create function to do countplot of different features vs Heart Disease
def cplot(col):
    plt.figure()
    sns.countplot(x=col, hue='HeartDisease', data=heart_disease, palette='RdBu')
    plt.xticks(list(range(heart_disease[col].unique().size)), list(heart_disease[col].unique()))
    plt.show()

for i in cat_data:
    cplot(i)
num_data_cols = num_data.columns.values
num_data_cols = np.append(num_data_cols,'HeartDisease')
print(num_data_cols)
sns.pairplot(heart_disease.loc[:,num_data_cols], hue='HeartDisease')
plt.show()

# To use scikit learn we need to encode categorical features - use get_dummies()
heart_disease_encoded = pd.get_dummies(heart_disease, drop_first=True)
print(heart_disease_encoded.head())
print(heart_disease_encoded.info())


# Create feature and target arrays
X = heart_disease_encoded.drop('HeartDisease_1', axis=1)
y = heart_disease_encoded['HeartDisease_1']
print(X)
print(y)
#y = heart_disease_encoded['HeartDisease_1'].values()
#print(y)
#_ = pd.plotting.scatter_matrix(X, c = y, marker = 'D')


#


