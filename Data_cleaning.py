import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils import resample


#Step 1: prepare data
def data_prepration (data):
    df1 =data.copy()
    #Clean data
    #Replace missing values with zero
    df1['V4'] = df1['V4'].fillna(0)

    #Filter Time column
    df1 = df1 [(df1['Time'] != 0) & (df1['Time'] >= 2)] 

    # Convert categorical data to numerical data
    encoder = LabelEncoder()
    df1['Class'] = encoder.fit_transform(df1['Class'])
    return df1

#Step 2: Train and Test Data
def train_test (data):
    df1 =data.copy()
    #Features
    X = df1 [['Time', 'Amount']+\
            ['V'+str(x) for x in range(1,29)]]
    #Output
    Y = df1 [['Class']]

    #Normalize data
    scaler = MinMaxScaler()
    scaler = scaler.fit(X)
    X = scaler.transform(X)  

    #Train and Test data
    x_train, X_test, y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, shuffle=True)
    df2 = pd.DataFrame(np.concatenate([x_train, y_train], axis=1), columns=df1.columns)



    #OverSample data
    def OverSample ():
        df_majority = df2[df2.Class==0]
        df_minority = df2[df2.Class==1]
        df_minority_Upsampled = resample(df_minority, 
                                        replace=True,      # sample with replacement
                                        n_samples=df_majority.shape[0])     # to match majority class
                                        
        # Combine minority class with downsampled majority class
        df_upsampled = pd.concat([df_minority_Upsampled, df_majority])
        Y_train = df_upsampled.Class.values.ravel()
        X_train = df_upsampled.drop('Class', axis=1)
        # summarize the new class distribution
        counter = Counter(Y_train)
        print(counter)
        return X_train, Y_train 


    #DownSample majority class
    def DownSample ():
        df_majority = df2[df2.Class==0]
        df_minority = df2[df2.Class==1]
        df_majority_downsampled = resample(df_majority, 
                                        replace=False,      # sample without replacement
                                        n_samples=df_minority.shape[0])     # to match minority class
                                        
        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
        Y_train = df_downsampled.Class.values.ravel()
        X_train = df_downsampled.drop('Class', axis=1)
        # summarize the new class distribution
        counter = Counter(Y_train)
        print(counter)
        return X_train, Y_train 

    #Create synthetic data
    def Synthetic_data ():
        oversample = SMOTE ()
        X_train, Y_train = oversample.fit_resample(x_train, y_train)
        # summarize the new class distribution
        value_counts = Y_train.value_counts()
        #print(value_counts)
        return X_train, Y_train

    X_train, Y_train = DownSample ()
    return X_train, Y_train, X_test, Y_test
