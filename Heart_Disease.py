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

# Summarize the basic information about the data using info() method:
print('\nSUMMARY OF FILE: \n', heart_disease.info(), '')

# Display the column labels to check if the column headers are in correct format using the columns attribute:
print('\nCOLUMNS LIST AND COUNT: \n', heart_disease.columns, '')

# Get a summary about the data with all the basic summary statistics using the describe() method for each field:
print('\nDESCRIPTION OF DATA\n\nAGE: \n Sample data: ', heart_disease['Age'][0], '\n', heart_disease['Age'].describe())
print('\nSex: \n Sample data: ', heart_disease['Sex'][0], '\n', heart_disease['Sex'].describe())
print('\nChest Pain Type: \n Sample data: ', heart_disease['ChestPainType'][0], '\n', heart_disease['ChestPainType'].describe())
print('\nResting Blood Pressure: \n Sample data: ', heart_disease['RestingBP'][0], '\n', heart_disease['RestingBP'].describe())
print('\nCholesterol: \n Sample data: ', heart_disease['Cholesterol'][0], '\n', heart_disease['Cholesterol'].describe())
print('\nFasting Blood Sugar: \n Sample data: ', heart_disease['FastingBS'][0], '\n', heart_disease['FastingBS'].describe())
print('\nResting ECG: \n Sample data: ', heart_disease['RestingECG'][0], '\n', heart_disease['RestingECG'].describe())
print('\nMax Heart Rate: \n Sample data: ', heart_disease['MaxHR'][0], '\n', heart_disease['MaxHR'].describe())
print('\nExercise Angina: \n Sample data: ', heart_disease['ExerciseAngina'][0], '\n', heart_disease['ExerciseAngina'].describe())
print('\nOldpeak: \n Sample data: ', heart_disease['Oldpeak'][0], '\n', heart_disease['Oldpeak'].describe())
print('\nST Slope: \n Sample data: ', heart_disease['ST_Slope'][0], '\n', heart_disease['ST_Slope'].describe())
print('\nHeart Disease: \n Sample data: ', heart_disease['HeartDisease'][0], '\n', heart_disease['HeartDisease'].describe())


##------------------------------------------
# Step 3 - Cleaning and Formatting the data
##------------------------------------------

# Convert Fasting Blood Sugar from numerical to categorical (1 = True, 0 = False)
heart_disease['FastingBS'] = heart_disease['FastingBS'].astype('category')
assert heart_disease['FastingBS'].dtype == 'category'
print('\nRevised Fasting Blood Sugar: \n Sample data: ', heart_disease['FastingBS'][0], '\n', heart_disease['FastingBS'].describe())

# Convert Heart Disease from numerical to categorical (1 = True, 0 = False)
heart_disease['HeartDisease'] = heart_disease['HeartDisease'].astype('category')
assert heart_disease['HeartDisease'].dtype == 'category'
print('\nHeart Disease: \n Sample data: ', heart_disease['HeartDisease'][0], '\n', heart_disease['HeartDisease'].describe())

# Check for any Missing Values (NaN)
print('\nIDENTIFY MISSING VALUES: \n', heart_disease.isnull().sum(), '')

# Remove duplicates and re-chack shape
heart_disease.drop_duplicates(inplace=True)
print('\nDATA SHAPE AFTER REMOVING DUPLICATES: \n', heart_disease.shape, '')



