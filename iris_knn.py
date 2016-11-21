"""
In 1936, Ronald Fisher presented a model to discriminate between 3
species of Iris' based on 4 predictors the lenght and width of the sepals
and the petals. Here we will use the k-nearest neighbors model to predict
the species. The data can be found on the UCI machine learning site:
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

In KNN we are trying to approximate P(Y=j|X=x0). That is given an input
vector X=x0 (predictors) what is the Probability the the result is in class
j? We approximate this distribution by getting the classes of
the k-nearest neighbors and assigning X=Xo the most likely one. Here we
are going to find the k-nearest neighbors using euclidean distances.
"""

import csv
import random
import math

from collections import Counter
from operator import itemgetter
from matplotlib import pyplot as plt

###################
# Import the data #
###################
# In iris.data the first 4 columns are the predictors and the last column 
# is the species (label). We will store these as (point,label) tuples.

def csv_to_point_label(csv_file, predictor_cols, label_col):
    """
    opens a csv file and creates 'data' a list of tuples (predictors,label)
    for each row in csv_file. Predictor cols specifies the columns
    containing the predictors and label_col specifies the label column.
    """
    data = []
    with open(csv_file,'r') as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            # convert predictors to floats
            predictors = [float(row[i]) for i in predictor_cols]
            # get the label
            label = row[label_col]
            # append tuple
            data.append((predictors,label))

    return data


##################
# Split the data #
##################
# We now split the data into a training set and a testing set
def split_train_test(data, ratio=0.67):
    """ splits data list into training and test examples using ratio """
    train = []
    test = []

    for _, data_ls in enumerate(data):
        if random.random() < ratio:
            train.append(data_ls)
        else:
            test.append(data_ls)

    return train, test


##########################
# Define distance metric #
##########################
def euclid_dist(point1, point2):
    """ computes the euclidean distance between two points """
    return math.sqrt(sum([(x1-x2)**2 for x1,x2 in zip(point1,point2)]))

#################
# Get Neighbors #
#################
def get_neighbors(training_set, labeled_point, k):
    """ 
    get the k nearest neighbors to (point,label) tuple from trainingSet
    of (points,label) tuples. 
    """

    def point_distance(training_point):
        """ 
        calculates the distance between a training_point and a
        labeled_point. 
        """
        return euclid_dist(training_point[0], labeled_point[0])

    # now get all the neighbors
    neighbors = sorted(training_set, key=point_distance)

    # k-nearest neighbor labels
    k_nearest_labeled_points = neighbors[0:k]

    return k_nearest_labeled_points

##########################
# Get the majority label #
##########################
def get_majority_label(labeled_points):
    """ 
    returns the label with the most counts from a list of (point,label)
    tuples.
    """

    # get the labels from the labeled points
    labels = [label for _,label in labeled_points]
        
    # get the most common label(s) note could be multiple winners
    winning_point_labels = Counter(labels).most_common()

    # if multiple winners randomly select winner
    if len(winning_point_labels) > 1:
        winning_label = random.choice(winning_point_labels)[0]
        
    # else winner is most common winner
    else:
        winning_label = winning_point_labels[0][0]

    return winning_label

###############
# Get k-value #
###############
# To determine the optimum k-value we will calculate the prediction error
# rate on the test_set for a variety of k-values and choose the k-value that
# minimizes the test error rate.

def get_optimum_k(training_data, test_data, k_values=list(range(1,11))):
    """
    Determines the k_value that minimizes the test_error rate in the KNN
    model using the training_data for class prediction.
    """
    k_error_rates = []
    for k in k_values:
        error_rate = 0
        for labeled_point in test_data:
            # for each laeled point in the test_data we get its neigbors and
            # the majority label
            neighbors = get_neighbors(training_data, labeled_point, k)

            predicted_label =  get_majority_label(neighbors)
            
            if predicted_label != labeled_point[1]:
                error_rate += 1/float(len(test_data))

        k_error_rates.append((k, error_rate))

    # now sort the k_percent_corrects and get the lowest error rate k
    optimum_k, min_error  = sorted(k_error_rates,key=itemgetter(1))[0]
    
    return optimum_k, min_error, k_error_rates
            
    
def main():
    """
    Performs KNN classification on train and test sets split from the iris 
    dataset. Attempts to locate an optimal k-value determined from the test 
    data.
    """
    # Get data from iris data file
    data = csv_to_point_label('__data__/iris.data', [0,1,2,3], 4)

    # split the data into train and test sets
    train_set, test_set = split_train_test(data, 0.67)

    # determine the k-value that minimizes the test error rate
    optimal_k, min_error, k_error_rates = get_optimum_k(train_set, test_set, 
                                             k_values=list(range(1,11)))
    
    print(optimal_k)


    print('For k=',optimal_k,'the percentage of correct test',
          'predictions is', round(1-min_error,3),'for', len(test_set), 
          'test trials')

    answer = input('Would you like to see the error'\
                    ' rates as a function of k? ')

    if answer in {'Y','y','Yes','yes'}:
        # show user plot of error rates
        plt.plot(*zip(*k_error_rates), linestyle='--',marker='o', 
                 markersize=10);
        plt.xlabel('K-Neighbors')
        plt.ylabel('Error rate')
        plt.show()

main()

