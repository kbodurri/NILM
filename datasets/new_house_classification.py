#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

pd.options.mode.chained_assignment = None

# A class to encode appliances!
class LabelEncoder(object):

    def __init__(self):
        self.encoding = {}
        self.decoding = {}

    # Input: pandas dataframe and the column to encode, returns the a numpy array
    def apply_encoding(self, df):
        df.replace(self.encoding, inplace = True)
        return df.values
    
    # Input: the key, returns the decoded appliance
    def apply_decoding(self, key):
        return self.decoding[key]

    # Input: pandas dataframe and the column t create the uniques code per appliance! (e.g daskio)
    def create_encoding_decoding(self, df):
        custom_id = 1
        for appliance in df.unique():
            self.decoding[custom_id] = appliance
            self.encoding[appliance] = custom_id
            custom_id += 1

def data_process(df):
    features = ['I50', 'Φ50', 'I150', 'Φ150', 'I250', 'Φ250'] # 50-150-250Hz
    
    # Data Cleaning with Aggelos Rules for 50, 150, 250 phases
    df = df[(df.I50 > 0.1) & (df.I150 > 0.01) & (df.I250 > 0.01)] # Clean useless current features

    # For angle between (90, 180):
    # Modify by +180 degrees
    rows_with_rads_to_decrease = df.loc[(df['Φ50'] > 90) & (df['Φ50'] < 180)]
    rows_with_rads_to_decrease['Φ50'] -= 180
    df.update(rows_with_rads_to_decrease)

    # For angle between (-180, -90):
    # Modify by -180 degrees
    rows_with_rads_to_increase = (df.loc[(df['Φ50'] < -90) & (df['Φ50'] > -180)])
    rows_with_rads_to_increase['Φ50'] += 180
    df.update(rows_with_rads_to_increase)
    
    # Calculate Z-score in order to find outliers
    z = np.abs(stats.zscore(df[features]))
    #print(z) # Visualize
    threshold = 2.5 # Change the threshold arbitrarily
    #print(np.where(z > threshold))
    df = df[(z < threshold).all(axis=1)] # Remove outliers that exceed the threshold given from dataset
    
    ####-_-_-###-_-_-####-_-_-###-_-_-####-_-_-###-_-_-####-_-_-###-_-_-
    #removeColumns = ['I150', 'Φ150', 'I250', 'Φ250'] # 50
    #remainingColumns = ['I50', 'Φ50']
    
    #removeColumns = ['I250', 'Φ250'] # 50-150
    #remainingColumns = ['I50', 'Φ50', 'I150', 'Φ150']
    
    #df = df.drop(removeColumns, axis=1)
    ####-_-_-###-_-_-####-_-_-###-_-_-####-_-_-###-_-_-####-_-_-###-_-_-####-_-_-###-_-_-
    remainingColumns = ['I50', 'Φ50', 'I150', 'Φ150', 'I250', 'Φ250']
    
    # Label - Y
    y = le.apply_encoding(df['appliance'])     
    
    # Now get as X the 'clean' features
    X = df[remainingColumns]
    # -3- Robust scaling
    scaler = preprocessing.RobustScaler()
    X = scaler.fit_transform(X)
    
    return X, y


# create an instance of label encoder!
le = LabelEncoder()

# Uncomment the following if you want to train on house1 and test on house2:
#daskio = pd.read_excel("appliances_combination_daskio.xls")
#veroia = pd.read_excel("appliances_combination_veroia.xls")

# make daskio and veroia have the same classes
#unique_veroia = veroia['appliance'].unique()
#daskio = daskio[daskio['appliance'].isin(unique_veroia)]

#init encoding and decoding from daskio dataset!
#le.create_encoding_decoding(daskio['appliance'])
#X_train, y_train = data_process(veroia)
#X_test, y_test = data_process(daskio)
######----#####------######----#####------######----#####------######----#####------

# Train & Testing in 1 dataset only
daskio = pd.read_excel("../datasets/appliances_combination_daskio.xls")

# init encoding and decoding from daskio dataset!
le.create_encoding_decoding(daskio['appliance'])

# get x,y from daskio
X, y = data_process(daskio)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def printMetrics(clf, X_test, y_test):
    # This will print precision, recall, f1-score, support for all the categories
    #target_names = ['class 0', 'class 1', 'class 2']
    print("Classification report for classifier \n%s:\n%s" % (clf, metrics.classification_report(y_test, y_pred)))
    print("Accuracy: %1.3f" % clf.score(X_test, y_test))
    print("-----------------\n")   

# Compute the metrics mathematically 
def get_metrics(true_string, predicted_string):
    
    true_array = true_string.split("+") # Find substrings and split them
    predicted_array = predicted_string.split("+")
    
    # Find which elements of array A are in array B
    comparePositives = np.in1d(predicted_array, true_array)
    compareNegatives = np.in1d(true_array, predicted_array)
    
    TP = TN = FP = FN = 0 # Initialize the metrics
    
    # Scan the arrays for each elements' presence
    for predictedLabel in comparePositives:
        if (predictedLabel == True):
            TP = TP + 1
        if (predictedLabel == False):
            FP = FP + 1
            
    for predictedLabel in compareNegatives:
        if (predictedLabel == True):
            TN = TN + 1
        if (predictedLabel == False):
            FN = FN + 1

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    
    if (precision+recall == 0):
        F1 = 0
    else:
        F1 = 2*((precision*recall)/(precision+recall))
    
    return precision, recall, F1

# Decision Trees
clf = DecisionTreeClassifier(random_state = 42) # Feel free to change 'min_samples_split' 
clf.fit(X_train, y_train)

#print("Decision Trees:")
y_pred = clf.predict(X_test)

printMetrics(clf, X_test, y_test)

# Initialiaze a list
list_of_classes = []

# Populate the list with all the class names
for i in range(0, len(y_test)):
    true_class = le.apply_decoding(y_test[i])
    list_of_classes.append(true_class)

# Keep only non-duplicated class names
list_of_classes = sorted(set(list_of_classes), key=lambda x: list_of_classes.index(x))

# Initialiaze a dictionary thas has as 'key' the class name and as 'values' the precision, recall, f1 metrics
mean_dict = {class_name:[0, 0, 0, 0] for class_name in list_of_classes}

print(mean_dict)


# In[13]:


print(len(mean_dict))


# In[14]:


for i in range(0, len(y_test)):

    true_class = le.apply_decoding(y_test[i])
    predicted_class = le.apply_decoding(y_pred[i])

    print('Actual class is ' + color.BOLD  + color.BLUE+ true_class + color.END +' and the classifier predicted ' + color.BOLD + color.RED + predicted_class + color.END)
    precision, recall, F1 = get_metrics(true_class, predicted_class)
    
    print('Precision is {}, Recall is {} and F1-Score is {}'.format(precision, recall, F1))
    
    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = round(F1, 4)
    
    # Add the values to the dictionary so we can have mean-values for the metrics
    mean_dict[true_class][0] = mean_dict[true_class][0] + precision 
    mean_dict[true_class][1] = mean_dict[true_class][1] + recall
    mean_dict[true_class][2] = mean_dict[true_class][2] + F1
    mean_dict[true_class][3] = mean_dict[true_class][3] + 1
    print('\n')
    

for key in mean_dict:
    count = mean_dict[key][3]
    mean_dict[key][0] /= count
    mean_dict[key][1] /= count
    mean_dict[key][2] /= count
    
output_df = pd.DataFrame(mean_dict)
output_df = output_df.T
output_df.to_csv('daskio_decision_trees_50_150_250.csv')
