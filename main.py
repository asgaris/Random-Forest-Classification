import ML_Models as myRF 
import Data_cleaning 
import pandas as pd
import matplotlib.pyplot as plt


def getData(data):
    X_train, Y_train, X_test, Y_test = Data_cleaning.train_test(data)
    return {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test':Y_test}


#Data from: https://www.kaggle.com/adhyanmaji31/credit-card-fraud-detection
data = pd.read_csv(r'Path to data\credit_card_fraud_detection.csv')
cleandata = Data_cleaning.data_prepration (data)
datadict = getData(cleandata)

#print (datadict)

results = myRF.Random_forest( X_train= datadict['X_train'], Y_train=datadict['Y_train'], X_test = datadict['X_test'], Y_test=datadict['Y_test'])
