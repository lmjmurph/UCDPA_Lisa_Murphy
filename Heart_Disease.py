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

# Look closer at Cholesterol values - min value = 0 doesn't make sense
print(heart_disease['Cholesterol'].value_counts().sort_index())
heart_disease['Cholesterol'].replace([0],np.nan, inplace=True)

describe_all(heart_disease)


sns.boxplot(x="Sex", hue="HeartDisease", y="Cholesterol", data=heart_disease)
# adding a title to the plot:
plt.title('Renme', fontsize=14)
# displaying the plot:
plt.show()




