import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def plot_categorical(df, **kwargs) -> None:
    """
    this function plots all categorical variables count plots in pie style
    
    Arguments:
    df -- dataframe
    kwargs -- only categorical variables
    
    Returns:
    None
    """
    num_row = math.ceil(len(kwargs)/3)
    
    f = plt.subplots(num_row, 3, figsize = (20,20))
    
    idx = 1
    for key, value in kwargs.items():
        plt.subplot(num_row, 3, idx)
        df[key].value_counts().plot.pie(title=value, autopct="%1.1f")
        idx += 1

    plt.show()
    
    
def cat_with_output(df, hue="Survived", **kwargs) -> None:
    """
    this function plots all categorical variables count and also separate it with hue
    
    Arguments:
    df -- dataframe
    hue -- A feature that sets our plot apart
    kwargs -- only categorical variables
    
    Returns:
    None
    """
    num_row = math.ceil(len(kwargs)/2)
    
    f, ax = plt.subplots(num_row, 2, figsize = (25,10))
    
    idx = 1
    for key, value in kwargs.items():
        plt.subplot(num_row, 2, idx)
        sns.countplot(x=key, data=df, hue=hue)
        idx +=1
        
    plt.show()
    
    
def convert_age(age: int, *args) -> int:
    """
    this function gets age and age ranges returns the index depending on which range it falls into
    
    Arguments:
    age -- which we want to convert for examle child yound old adults  
    args -- age ranges 
    
    Returns:
    index based on ranges
    """
    age_ranges = args
    
    for i in range(len(age_ranges)):
        if age < age_ranges[i]:
            return i
    
    return len(age_ranges)


def featureNormalization(df, features) ->None:
    """
    this function normilize features by standartScaler library
    
    Arguments:
    features -- array of strings
    
    Returns:
    None
    """
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(df[features])
    scaler_df = pd.DataFrame(scaled_df, columns=features)
    df[features] = scaled_df
    

def get_encoded_dict(df, lst):
    """
    this function creates dictionary for encoding. Its find unique labels for each column and enumerate them
    
    Arguments:
    df -- pandas dataframe
    lst -- list of columns which we want to encode 
    
    Returns:
    dictionary where key is column name and value is dictionary of unique labels and encoding value
    """
    encoded_dict = {}
    for col in lst:
        each_dict = {}
        sorted_unique_names = df[col].dropna().unique()
        sorted_unique_names.sort()
        for i,val in enumerate(sorted_unique_names):
            each_dict[val] = i
        encoded_dict[col] = each_dict
    return encoded_dict