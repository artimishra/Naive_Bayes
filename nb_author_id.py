
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.naive_bayes import GaussianNB

    ### create classifier
clf = GaussianNB()

    ### fit the classifier on the training features and labels
    #TODO
t0 = time()
clf.fit(features_train,labels_train)
print ('training time', round(time()-t0, 3), "s")

    ### use the trained classifier to predict labels for the test features
t1 = time()
pred = clf.predict(features_test)
print ('predict time:', round(time()-t1, 3), "s")
    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
from sklearn.metrics import accuracy_score 
accuracy= accuracy_score(pred,labels_test)
print(accuracy)



#########################################################


