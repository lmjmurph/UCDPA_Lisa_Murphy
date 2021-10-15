
### Analysis of Irish publications from Elsevier's Scival

## Step 1 - Importing relevant Python libraries for analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


## Step 2 - Load publications data into a pandas dataframe and get an overview of the data

publications = pd.read_csv('C:/Users/lisa.murphy/Desktop/To be deleted/publications-1.csv')
print('First 5 rows of Data:', publications.head(), '')
# Studying the data size (total rows & columns) using the shape attribute:
print('Data shape:', publications.shape, '')
# Summarizing the basic information about the data using info() method:
#print('File information:', publications.info(), '')
# Displaying the column labels to check if the column headers are in correct format using the columns attribute:
print('List of Columns:', publications.columns, '')
# Getting a summary about the data with all the basic summary statistics using the describe() method:
print('Description of data: ', publications.describe())

## Step 3 - Cleaning and Formatting the data

# Checking for any Missing Values (NaN)
print('Identify missing values:', publications.isnull().sum(), '')

# Remove duplicates and re-chack shape
publications.drop_duplicates(inplace=True)
print('Data shape:', publications.shape, '')

# check the publication type categories are consistent and count how many records fall into each category
print(publications['Publication type'].value_counts())

# categorise citation categories
print('Description of FWCI data: ', publications['Field-Weighted Citation Impact'].describe())

# Create ranges for categories
label_ranges = [0, 1, 2, np.inf]
label_names = ['low', 'medium', 'high']
# Create citation category column
publications['citation_cat'] = pd.cut(publications["Field-Weighted Citation Impact"], bins = label_ranges,
                                labels = label_names)

# check the publication type categories are consistent and count how many records fall into each category
print(publications['citation_cat'].value_counts())
###### ONLY SELECT ARTICLE, REVIEW, CONFERENCE PAPER, CHAPTER, BOOK

# Imputing missing values with the mean of the column using the fillna() and mean() functions:
#life_data.fillna(value = life_data.mean(), inplace = True)
# Re-calculating the basic summary statistics of dataset post-imputation of missing values:
#print(publications.describe())
# Re-checking for sum of Missing values in each column after imputation:
#print(publications.isnull().sum())
#_ = sns.boxplot(x="Publication type", y ="Field-Weighted Citation Impact", data = publications)
#_ = sns.boxplot(data=publications, x='Year', y='Field-Weighted Citation Impact')
#_ = plt.xlabel('FWCI')

#plt.show()

_ = plt.hist(publications['Field-Weighted Citation Impact'], bins=200)
_ = plt.xlabel('Birth weight (lb)')
_ = plt.ylabel('Fraction of births')
plt.show()