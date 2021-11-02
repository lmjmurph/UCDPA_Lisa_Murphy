
### Analysis of Irish publications from Elsevier's Scival

## Step 1 - Importing relevant Python libraries for analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import boxcox



## Step 2 - Load publications data into a pandas dataframe and get an overview of the data

publications_1 = pd.read_csv('C:/Users/lisa.murphy/Desktop/To be deleted/Publications- 2016-2018.csv')
# Check shape of data
print(f"Publications_1 data shape:{publications_1.shape}")

publications_2 = pd.read_csv('C:/Users/lisa.murphy/Desktop/To be deleted/Publications-2019-2020.csv')
# Check shape of data
print(f"Publications_2 data shape:{publications_2.shape}")

# Merge datasets
publications = pd.DataFrame.merge(publications_1, publications_2, how="outer")
print(f"Merged dataframe: {publications.shape}")

# set pd display options to allow view of all columns
pd.options.display.width= None
pd.options.display.max_columns= None

# display first 5 rows
print(f"\nFirst 5 rows of Data:\n{publications.head()}")

# Summarizing the basic information about the data using info() method:
print("\nFile information:")
print(publications.info())

# Getting a summary about the data with all the basic summary statistics using the describe() method:
print(f"Description of data: \n{publications.describe(include='all')}")

##------------------------------------------
# Step 3 - Cleaning and Formatting the data
##------------------------------------------

# Checking for any Missing Values (NaN)
print('Identify missing values:', publications.isnull().sum(), '')

# Remove duplicates and re-chack shape
publications.drop_duplicates(inplace=True)
print('Data shape:', publications.shape, '')

# check the publication type categories are consistent and count how many records fall into each category
print(publications['Publication type'].value_counts())

# Convert Year field to Category Type
publications['Year'] = publications['Year'].astype('category')
print('REVISED YEARS', publications['Year'].describe())

# Select only records that have publication type Article, Conference Paper, Review, Chapter
# Other types are no relevant to this analysis

rel_pubs = {'Article', 'Review', 'Conference Paper', 'Chapter','Book'}
publications = publications.loc[publications['Publication type'].isin(rel_pubs)]
print(publications.shape)
print(publications.columns)

# categorise citation categories
print('Description of FWCI data: ', publications['Field-Weighted Citation Impact'].describe())

# Create ranges for categories
label_ranges = [0, 0.8, 1.2, 3, np.inf]
label_names = ['low', 'average', 'above average', 'high']
# Create citation category column
publications['citation_cat'] = pd.cut(publications["Field-Weighted Citation Impact"], bins = label_ranges,
                                labels = label_names)

# check the publication type categories are consistent and count how many records fall into each category
print(publications['citation_cat'].value_counts())

# create ecdf
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

# Compute ECDF for FWCI data: x_fwci, y_fwci
x_fwci, y_fwci = ecdf(publications['Field-Weighted Citation Impact'])




# Generate plot
_ = plt.plot(x_fwci, y_fwci, marker = '.', linestyle = 'none')

# Label the axes
_ = plt.xlabel('FWCI')
_ = plt.ylabel('ECDF')


# Display the plot
plt.show()


# try box cox transform
publications['bc_fwci'] = boxcox(publications['Field-Weighted Citation Impact']+1, 0)
_ = plt.scatter(x="Year", y='bc_fwci', data = publications)
_ = plt.xlabel('Year')
_ = plt.ylabel('Bco_Cox FWCI')
plt.show()

sns.boxplot(x='Year', y='Field-Weighted Citation Impact', data=publications, whis=10)
plt.yscale('log')
plt.show()

###### ONLY SELECT ARTICLE, REVIEW, CONFERENCE PAPER, CHAPTER, BOOK

#print(publications.isnull().sum())
_ = sns.boxplot(x="Field-Weighted Citation Impact", y="International collab", data = publications)
#_ = sns.boxplot(data=publications, x='Year', y='Field-Weighted Citation Impact')
_ = plt.xlabel('FWCI')
plt.show()

def normalize(column):
    upper = column.max()
    lower = column.min()
    y = (column - lower)/(upper-lower)
    return y

# FWCI data highly skewed - try normalise using log (plus 1 to avoid zero divisor)
publications['FWCI_log'] = np.log(publications['Field-Weighted Citation Impact']+1)
publications['FWCI_log_normalized'] = normalize(publications['FWCI_log'])
print(publications['FWCI_log_normalized'].describe())

_ = plt.hist(publications['bc_fwci'], bins=20)
_ = plt.xlabel('bc_FWCI')
_ = plt.ylabel('No. of publications')
plt.show()

_ = plt.scatter(x="Year", y='Field-Weighted Citation Impact', data = publications)
_ = plt.xlabel('Year')
_ = plt.ylabel('FWCI')
plt.show()




# Chart 1 - Visualizing FWCI by year
# creating the violin plot using seaborn (alias as sns) with FWCI on y-axis
sns.violinplot(x="Industry Collaboration", y="FWCI_log", data=publications,palette='rainbow')
# adding a title to the plot:
plt.title('FWCI (log) for different publication types')
# displaying the plot:
plt.show()

#use REGEX to filter the dataframe for publications about Heart Disease

heart_pubs = publications[publications['Title'].str.contains(".*[Hh]eart\s[Dd]isease.*")==True]
print(f"Publications on Heart Disease:\n{heart_pubs.shape}")
print(heart_pubs.head())