import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


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


def classification_metrics(yTrueTrain, yPredictTrain, yTrueTest, yPredictTest, heatmap=False) -> None:
    """
    this function prints accuracy, precision, recall, F1 scre 
    and create confusion matrix  heatmap for both train and test sets
    
    Arguments:
    yTrueTrain -- the true value of prediction from the train set
    yPredictTrain -- predicted value from the train set
    yTrueTest -- he true value of prediction from the test set
    yPredictTest -- predicted value from the test set
    heatmap -- boolean which tells us if plot heatmap or not
    
    Returns:
    None
    """
    print("train data:\t  "+ "\t"*6+ "test data:\t\n")
    print(("accuracy:\t {0} "+ "\t"*6+ "accuracy:\t {1}").format(accuracy_score(yTrueTrain, yPredictTrain).round(2), 
                                                                 accuracy_score(yTrueTest, yPredictTest).round(2)))
          
    print(("precision:\t {0} "+ "\t"*6+ "precision:\t {1}").format(precision_score(yTrueTrain, yPredictTrain).round(2), 
                                                                   precision_score(yTrueTest, yPredictTest).round(2)))
    
    print(("recall:\t\t {0} "+ "\t"*6+ "recall:\t\t {1}").format(recall_score(yTrueTrain, yPredictTrain).round(2), 
                                                                 recall_score(yTrueTest, yPredictTest).round(2)))
    
    print(("F1:\t\t {0} "+ "\t"*6+ "F1:\t\t {1}").format(f1_score(yTrueTrain, yPredictTrain).round(2), 
                                                         f1_score(yTrueTest, yPredictTest).round(2)))
    if(heatmap):
        f, ax  = plt.subplots(1,2,figsize = (18,6))
        sns.heatmap(confusion_matrix(yTrueTrain,yPredictTrain),cmap='coolwarm',annot=True,ax=ax[0])
        sns.heatmap(confusion_matrix(yTrueTest,yPredictTest),cmap='coolwarm',annot=True,ax=ax[1])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        

def get_accuracy_based_on_depth(depth_range,X_train,y_train,cv):
    accuracies = []
    for depth in depth_range:
        fold_accuracy = []
        tree_model = DecisionTreeClassifier(max_depth = depth,random_state = 42)

        for train_fold, valid_fold in cv.split(X_train):
            f_train_x = X_train.iloc[train_fold] 
            f_train_y = y_train.iloc[train_fold]

            f_valid_x = X_train.iloc[valid_fold]
            f_valid_y = y_train.iloc[valid_fold]

            model = tree_model.fit(X = f_train_x,y = f_train_y) # We fit the model with the fold train data
            valid_acc = model.score(X = f_valid_x,y = f_valid_y) # We calculate accuracy with the fold validation data
            fold_accuracy.append(valid_acc)

        avg = sum(fold_accuracy)/len(fold_accuracy)
        accuracies.append(avg)

    df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
    df = df[["Max Depth", "Average Accuracy"]]
    return df