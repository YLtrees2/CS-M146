
"""
Description : Titanic
"""

## IMPORTANT: Use only the provided packages!

## SOME SYNTAX HERE.   
## I will use the "@" symbols to refer to some variables and functions. 
## For example, for the 3 lines of code below
## x = 2
## y = x * 2 
## f(y)
## I will use @x and @y to refer to variable x and y, and @f to refer to function f

import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


######################################################################
# classes
######################################################################

class Classifier(object) :

    ## THIS IS SOME GENERIC CLASS, YOU DON'T NEED TO DO ANYTHING HERE. 

    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) : ## INHERITS FROM THE @CLASSIFIER

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        # n,d = X.shape ## get number of sample and dimension
        y = [self.prediction_] * X.shape[0]
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- an array specifying probability to survive vs. not 
        """
        self.probabilities_ = None ## should have length 2 once you call @fit

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        # in simpler wordings, find the probability of survival vs. not
        
        #The majority did not survive, which is Counter(y).most_common(1) 
        not_rate = Counter(y).most_common(1)[0][1]/sum(Counter(y).values())
        survival_rate = 1- not_rate
        self.probabilities_= [not_rate, survival_rate]
        
        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (check the arguments of np.random.choice) to randomly pick a value based on the given probability array @self.probabilities_

        y = np.random.choice([0,1], size=len(X), replace=True, p=self.probabilities_)

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use @train_test_split to split the data into train/test set 
    # xtrain, xtest, ytrain, ytest = train_test_split (X,y, test_size = test_size, random_state = i)
    # now you can call the @clf.fit (xtrain, ytrain) and then do prediction
    
    train_scores = []; test_scores = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done. 

    for i in range(0, 100):
        xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size = test_size, random_state = i)
        clf.fit(xtrain, ytrain)
        y_pred_train = clf.predict(xtrain)        # take the classifier and run it on the training data
        train_scores.append(1 - metrics.accuracy_score(ytrain, y_pred_train, normalize=True))
        y_pred_test = clf.predict(xtest)        # take the classifier and run it on the testing data
        test_scores.append(1 - metrics.accuracy_score(ytest, y_pred_test, normalize=True))
      
    train_error = sum(train_scores) / 100 ## average error over all the @ntrials
    test_error = sum(test_scores) / 100


    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier() # create Random classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion="entropy") 
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    # call the function @DecisionTreeClassifier

    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    clf = KNeighborsClassifier(n_neighbors=3) 
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error(k=3): %.3f' % train_error)
    
    clf = KNeighborsClassifier(n_neighbors=5) 
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error(k=5): %.3f' % train_error)
    
    clf = KNeighborsClassifier(n_neighbors=7) 
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error(k=7): %.3f' % train_error)
    # call the function @KNeighborsClassifier

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    # call your function @error

    clf = MajorityVoteClassifier() 
    train_error, test_error = error(clf, X, y)
    print('\t-- MajorityVote training error = %.3f, test error = %.3f' % (train_error , test_error))

    clf = RandomClassifier() 
    train_error, test_error = error(clf, X, y)
    print('\t-- RandomClassifier training error = %.3f, test error = %.3f' % (train_error , test_error))

    clf = DecisionTreeClassifier(criterion="entropy") 
    train_error, test_error = error(clf, X, y)
    print('\t-- DecisionTreeClassifier training error = %.3f, test error = %.3f' % (train_error , test_error))

    clf = KNeighborsClassifier(n_neighbors=5)
    train_error, test_error = error(clf, X, y)
    print('\t-- KNeighborsClassifier training error = %.3f, test error = %.3f' % (train_error , test_error))

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    # hint: use the function @cross_val_score
    k = list(range(1,50,2))
    cv_score = [] ## track accuracy for each value of $k, should have length 25 once you're done
    for i in k:
        ## YOU CONTINUE TO FINISH THIS PART. 
        clf = KNeighborsClassifier(n_neighbors=i) # create KNN classifier
        cv_score.append(np.average(cross_val_score(clf, X, y, cv=10)))


    print('\t-- The best value of k is %d' % (2*cv_score.index(max(cv_score))+1))
    error_rate = []
    for i in range(25):
        error_rate.append(1-cv_score[i])
    plt.plot(k, error_rate, marker='o')
    plt.xlabel('k')
    plt.ylabel('Error rate')
    plt.show()

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    depths = range(1, 21)
    train_errors = []
    test_errors = []
    for i in depths:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=i) 
        train_error, test_error = error(clf, X, y)
        train_errors.append(train_error)
        test_errors.append(test_error)
    plt.plot(depths, train_errors, marker='o', label='Training error')
    plt.plot(depths, test_errors, marker='o', label='Test error')
    plt.xlabel('Depth')
    plt.ylabel('Error rate')
    plt.legend()
    plt.show()
    
    print('\t-- The best value of depth is %d' % (test_errors.index(min(test_errors))+1))

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    
    train_errors_knn = []
    train_errors_dt = []
    test_errors_knn = []
    test_errors_dt = []
    x_axis = []
    
    #xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.1)

    for j in range(0, 100):

        xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.1, random_state=j)
        
        train_scores_knn = []; test_scores_knn = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done. 
        train_scores_dt = []; test_scores_dt = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done. 
    

        for i in range(1, 11):
            if i < 10:
                xtrain_train, xtrain_test, ytrain_train, ytrain_test = train_test_split(xtrain, ytrain, train_size = (i*0.1), random_state = j)
            else:
                xtrain_train = xtrain
                ytrain_train = ytrain
                
            clf_knn = KNeighborsClassifier(n_neighbors=7)
            clf_knn.fit(xtrain_train, ytrain_train)

            y_pred_train = clf_knn.predict(xtrain_train)        # take the classifier and run it on the training data
            train_scores_knn.append(1 - metrics.accuracy_score(ytrain_train, y_pred_train, normalize=True))
            y_pred_test = clf_knn.predict(xtest)        # take the classifier and run it on the testing data
            test_scores_knn.append(1 - metrics.accuracy_score(ytest, y_pred_test, normalize=True))
            
            clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=6)
            clf_dt.fit(xtrain_train, ytrain_train)

            y_pred_train = clf_dt.predict(xtrain_train)        # take the classifier and run it on the training data
            train_scores_dt.append(1 - metrics.accuracy_score(ytrain_train, y_pred_train, normalize=True))
            y_pred_test = clf_dt.predict(xtest)        # take the classifier and run it on the testing data
            test_scores_dt.append(1 - metrics.accuracy_score(ytest, y_pred_test, normalize=True))


        train_errors_knn.append(train_scores_knn) ## each will become 100*10
        test_errors_knn.append(test_scores_knn)
        train_errors_dt.append(train_scores_dt) ## each will become 100*10
        test_errors_dt.append(test_scores_dt)


        
    train_errors_knn_avg=[]
    test_errors_knn_avg=[]
    train_errors_dt_avg=[]
    test_errors_dt_avg=[]
    
    for i in range(1, 11):
        x_axis.append(i*0.1)
        train_knn_sum=0
        test_knn_sum=0
        train_dt_sum=0
        test_dt_sum=0
        for j in range(0,100):
            train_knn_sum += train_errors_knn[j][i-1]
            test_knn_sum += test_errors_knn[j][i-1]
            train_dt_sum += train_errors_dt[j][i-1]
            test_dt_sum += test_errors_dt[j][i-1]
        train_errors_knn_avg.append(train_knn_sum/100)
        test_errors_knn_avg.append(test_knn_sum/100)
        train_errors_dt_avg.append(train_dt_sum/100)
        test_errors_dt_avg.append(test_dt_sum/100)
        

    plt.plot(x_axis, train_errors_dt_avg, marker='o', label='DT Training Error')
    plt.plot(x_axis, test_errors_dt_avg, marker='o', label='DT Test Error')
    plt.plot(x_axis, train_errors_knn_avg, marker='o', label='KNN Training Error')
    plt.plot(x_axis, test_errors_knn_avg, marker='o', label='KNN Test Error')
    plt.xlabel('Proportion of training set used')
    plt.ylabel('Error rate')
    plt.legend(loc='lower right')
    plt.show()
    
    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
