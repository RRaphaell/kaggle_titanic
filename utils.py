import math
import matplotlib.pyplot as plt
import seaborn as sns

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