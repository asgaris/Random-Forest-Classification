from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
from matplotlib import pyplot
import numpy as np

# Create Random forest  model
def Random_forest (X_train, Y_train, X_test, Y_test):
    RF = RandomForestClassifier(max_depth=2, random_state=0)
    RF.fit(X_train, Y_train.ravel())
    test_pred  = RF.predict_proba(X_test)
    #print (testresults)
    testresults = getresults(test_pred[:,1], Y_test.astype('int'))
    return {'model': RF, 'results':testresults}


#Print recall, precision and f1score results
def getresults (prob_preds, y):
    fpr, tpr, thresholds = roc_curve(y, prob_preds)
    d = np.sqrt(np.add(np.square(1-tpr),np.square(fpr)))        
    ix = np.argmin(d)
    prob_preds = np.where(prob_preds > thresholds[ix],1,0)

    recall = recall_score(y, prob_preds)
    precision = precision_score(y,prob_preds)
    f1score = f1_score(y, prob_preds)
    matrix = confusion_matrix(y, prob_preds)
    ROC_AUC = roc_auc_score(y, prob_preds)
    print (f'Display Recall:\n {recall} \n')
    print (f'Display Precision:\n {precision} \n')
    print (f'Display F1score:\n {f1score} \n')
    print (f'Display confusion matrix:\n {matrix} \n')
    print (f'Display ROC_AUC:\n {ROC_AUC} \n')

    pyplot.plot(fpr, tpr, color='orange', linestyle='dotted', label='Deep Learning Model')
    pyplot.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='No-Skill Classifier')
    pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    return {'f1':f1score, 'precision':precision,'recall':recall, 'ROC_AUC':ROC_AUC, 'confusion matrix':matrix}
    