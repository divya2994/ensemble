# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 23:54:49 2019

@author: Divya Thakkar
"""

import math
import numpy as np
import pandas as pd
import random
import pprint
#Loading the dataset with comma separated values and giving names to the columns
data_iris=pd.read_csv("iris.data", sep=',',names=['sepal_length','sepal_width','petal_length','petal_width','label'])
#printing the dataset
data_lens=pd.read_csv("lens.data", delim_whitespace=True, names=['sr_no','age','spectacle_prescription','astigmatic','tear_production_rate','label'])
del data_lens['sr_no']

    
def split_dataset(df,test_size):
    random.seed(2)
    indices=df.index.tolist() # gives list of indices
    test_indices=random.sample(population=indices,k=test_size) #randomply sample k indices
    test_df=df.loc[test_indices] #subset of data(test) with test_indices values
    train_df=df.drop(test_indices) # training data with remaining indices(excluding the test indices)
    return train_df,test_df

def type_of_feature(df,feature):
    unique_values = np.unique(df[feature])# Extracting the unique values of a feature
    if len(unique_values)<=5 or isinstance(unique_values[0],str): # Check if the feature has less than 5 unique values or is type of string
        feature_type = 'categorical'
    else:
        feature_type = 'continuous'
    return feature_type

def entropy(df):
    label_column = df.values[:,-1] # accessing the label column of the dataset which is usually the last column
    unique_classes,count=np.unique(label_column,return_counts=True) # Extract unique labels and count of each unique label
    probability=count/count.sum()#numpy array of probabilities of each class
    total_entropy=0
    for i in probability:
        if i!=0:
            total_entropy+= i*-1*math.log2(i)# calculate the total entropy using the formula
        else:
            total_entropy+=0
    
    return total_entropy

def split_points(df,attr):
    splits=[]
    unique_values=np.unique(df[attr].values)#Extracting unique values
    if len(unique_values)<=1: #Checing if the dataframe has only 1 unique value we then return the value itself
        split = unique_values[0]
        splits.append(split)
    else:
        for i in range(len(unique_values)): 
            if i!=0:
                val1=unique_values[i] # Element 2
                val2=unique_values[i-1] # element 1
                split=(val1+val2)/2  # Average of adjacent elements 1 and 2
                splits.append(split)    
    return splits

def information_gain(df,orignal_dataframe,attr):
    total=0
    if type_of_feature(orignal_dataframe,attr)=='continuous':
        values=[] 
        value=split_points(df,attr) # finding split points for continuous attributes 
        for i in range(len(value)):
            x1=df.where(df[attr]<=value[i]).dropna() # subset with value less than or equal to split point
            x2=df.where(df[attr]>value[i]).dropna() # subset with value greater than split point
            probability_x1=len(x1)/len(x1+x2)
            probability_x2=len(x2)/len(x1+x2)
            total =(probability_x1*entropy(x1)+probability_x2*entropy(x2)) # calculating net entropy
            final=entropy(df) -total
            values.append(final) # appending the information gain values of each split point in the list names values
        index=np.argmax(values)# finding index of element with maximum information gain
        best_split_point = value[index] # finding value of element with maximum information gain
        best_split_gain = max(values) # findind maximum information gain value

    else: 
        value, count1=np.unique(df[attr],return_counts=True)
        total=0
        for i in range(len(value)):
            x=df.where(df[attr]==value[i]).dropna()
            final=count1[i]/count1.sum() # Array storing probabilities of each value of categorical variable
            total += final * entropy(x)
        best_split_gain=entropy(df) -total 
        best_split_point= None # setting split point to none as we do not consider any split point for categorical variable
       
    return best_split_point, best_split_gain

def bestfeature(df,orignal_dataframe,feature):
    gain_values=[]
    split_values=[]
    for features in feature:
        split_point,split_gain=information_gain(df,orignal_dataframe,features) # calculating information gain for every feature in list
        gain_values.append(split_gain) # Storing information gain values in list named gain_values
        split_values.append(split_point) # stroring split points corresponding to information gain values of feature
    best_feature_index=np.argmax(gain_values) # finding index of maximum information gain value
    best_feature=feature[best_feature_index] # finding feature corresponding to maximum gain value
    split_point=split_values[best_feature_index]
    #print('best feature is',best_feature,'with split point',split_point)
    return best_feature,split_point

def majority_label(data):
    label_column=data[:,-1]
    unique_labels,count_label=np.unique(label_column,return_counts=True)#Finding unique values and count of the unique values
    index=count_label.argmax()#Finding index of label with maximum count 
    majority_vote=unique_labels[index] #Finding label with maximum count
    return majority_vote

def decision_tree(dataset,orignal_dataset,features,feature_randomness=None):
    if len(np.unique(dataset['label'])) == 1:
        return np.unique(dataset['label'])[0] # return the label that the pure leaf contains
    if dataset.empty:
        return majority_label(orignal_dataset.values) # return the majority label of the original dataset if we get an empty dataset
    elif features ==[]:
        return majority_label(dataset.values) # return the majority label of parent node if we have no feature left
    else:
        no_of_features = len(dataset.columns[:-1].values) #Finding number of features in dataset excluding the label column
        feature_index_list = list(range(no_of_features))
        if feature_randomness and feature_randomness <= no_of_features: #check if feature_randomness argument is passed and its length is less than the number of features
            feature_indices = random.sample(population=feature_index_list, k=feature_randomness) #randomly selecting subset of feature indices as specified by k to build random forest
            features=[] 
            for feature_index in feature_indices:          
                feat = dataset.columns[ feature_index]
                features.append(feat)
            
        best_feature, value= bestfeature(dataset,orignal_dataset,features)#Use bestfeature function for finding the best feature
        #print("We use "+best_feature+" for splitting")
        tree = {best_feature:{}}
        subfeature =[]
        for i in features:
            if i!=best_feature:
                subfeature.append(i)
        if type_of_feature(orignal_dataset,best_feature)=='continuous':
            df_left=dataset.where(dataset[best_feature]<=value).dropna()#dataframe for left tree
            df_right=dataset.where(dataset[best_feature]>value).dropna()#dataframe for right tree
        
            value_left= '<='+str(value)
            value_right='>'+str(value)
        
            tree[best_feature][value_left]=decision_tree(df_left,orignal_dataset,subfeature) #tree for left subset(left child)
            tree[best_feature][value_right]=decision_tree(df_right,orignal_dataset,subfeature) # tree for right subset(right child)
        else:
            for value in np.unique(dataset[best_feature]):
            
                subset = dataset.where(dataset[best_feature] == value).dropna() # subtree for every category if the feature is categorical
                tree[best_feature][value] = decision_tree(subset,orignal_dataset,subfeature)
        return tree

def decision_stump(dataset,orignal_dataset,features):
    if len(np.unique(dataset['label'])) == 1:
        return np.unique(dataset['label'])[0] # return the label that the pure leaf contains
    if dataset.empty:
        return majority_label(orignal_dataset.values) # return the majority label of the original dataset if we get an empty dataset
    elif features ==[]:
        return majority_label(dataset.values) 
    else:
        best_feature, value= bestfeature(dataset,orignal_dataset,features)
        tree = {best_feature:{}}
        subfeature =[]
        if type_of_feature(orignal_dataset,best_feature)=='continuous':
            df_left=dataset.where(dataset[best_feature]<=value).dropna()#dataframe for left tree
            df_right=dataset.where(dataset[best_feature]>value).dropna()#dataframe for right tree
        
            value_left= '<='+str(value)
            value_right='>'+str(value)
        
            tree[best_feature][value_left]=decision_stump(df_left,orignal_dataset,subfeature)
            tree[best_feature][value_right]=decision_stump(df_right,orignal_dataset,subfeature)
        else:
            for value in np.unique(dataset[best_feature]):
            
                subset = dataset.where(dataset[best_feature] == value).dropna() # subtree for every category if the feature is categorical
                tree[best_feature][value] = decision_stump(subset,orignal_dataset,subfeature)
        return tree
    
def prediction(orignal_data,Tree,features,test,default_class=None):
    default_class= majority_label(orignal_data.values)
    class_Label= default_class
    a = list(Tree.keys())[0]#get the key from dictionary
    b = Tree[a] # accessing tree corresponding to key
    featureindex= features.index(a)
    if type_of_feature(orignal_data,features[featureindex])=='categorical':
        for key in b.keys():
            if test[featureindex]==key:
                if isinstance(b[key],dict):
                    return prediction(orignal_data,b[key],features,test) # call prediction function recursively until you get a leaf node
                else:
                    class_Label = b[key] # return the leaf node value
    else:
        key_list=list(b.keys())#list of keys for nested dictionary
        key_1=key_list[0]# value less than
        key_2=key_list[1]#value greater than
        key=key_1[2:]#
        if test[featureindex]<=float(key):
            ans=b[key_1] # refer subtree with value less than or equal
        else:
            ans=b[key_2] #refer subtree with value greater than or equal
        if isinstance(ans,dict): #call the recursive function until you get a class value
            return prediction(orignal_data,ans, features, test)
        else:   
            class_Label=ans
    return class_Label

def accuracy(testdf,orignal_data,tree):
    x=0
    for i in range(len(testdf)):
        ans =prediction(orignal_data,tree,testdf.columns[:-1].values.tolist(),testdf.values[i])
        #print('actual : '+str(ans)+ ' expected :' ,testdf['label'].values[i] )
        if ans==testdf['label'].values[i]:
            x+=1
        else:
            x+=0
    accuracy=(x/len(testdf))*100
    return accuracy

def bootstrap_samples(dataframe,number):
    samples=dataframe.sample(n=number, replace=True) #generating random samples with replacement
    return samples

def random_forest(df,original_dataset,no_of_trees,no_of_boot,no_of_features,features):
    forestlist=[]
    for i in range(no_of_trees):
        samples= bootstrap_samples(df,no_of_boot)
        tree =decision_tree(samples,original_dataset,features,feature_randomness=no_of_features)# building trees on bootstrapped samples
        forestlist.append(tree) #appending the trees in forestlist
        
    return forestlist

def random_predict(orignal_data,forest,test_vector):
    predictions=[]
    for i in range(len(forest)):
        predict = prediction(orignal_data,forest[i],orignal_data.columns[:-1].values.tolist(),test_vector) #making predictions using the prediction function
        predictions.append(predict)
    val,count=np.unique(predictions, return_counts=True)
    idx=count.argmax()
    majority=val[idx]
    return majority

def rand_accuracy(df,orignal_data,forest):
    x=0
    for i in range(len(df)):
        ans =random_predict(orignal_data,forest,df.values[i])
        if ans==df['label'].values[i]:
            x+=1
        else:
            x+=0
    accuracy=(x/len(df))*100
    return accuracy

if __name__ == '__main__':
    print("1.Implementation of decision trees, bagging and random forests for iris dataset \n2.Implementation of decision trees, bagging and random forests for lens dataset \n3.Implementation of decision stumps for iris and lens dataset ")
    x= int(input('enter choice'))
    if x==1:
        train_iris, test_iris = split_dataset(data_iris,50)
        # 50 here is test size. You can change this paremeter to values between 1 to 148 for testing as the dataset contains total 150 samples
        tree1 = decision_tree(train_iris,data_iris,train_iris.columns[:-1].values.tolist())
        accur1= accuracy(test_iris,data_iris,tree1)
        print("#######DECISION TREE FOR IRIS DATASET#########")
        pprint.pprint(tree1)
        print("The accuracy for iris dataset  using decision tree is: ", accur1)
        
        bagging_iris = random_forest(train_iris,data_iris,9,len(train_iris),4,train_iris.columns[:-1].values.tolist())
        ##3rd argument above is number of bagged trees. You can change the value for testing
        #To see the list of bagged trees remove the comment from below pprint statement
        #pprint.pprint(bagging_iris)
        bag_acur_iris = rand_accuracy(test_iris,data_iris,bagging_iris)
        print("The bagging accuracy for iris dataset is: ", bag_acur_iris)
        
        forest_iris = random_forest(train_iris,data_iris,23,len(train_iris),2,train_iris.columns[:-1].values.tolist())
        #3rd argument is number of trees.You can change the value for testing
        #5th argument is number of features which you can change between 1 to 4 for testing
        #To see the list of random forest remove the comment from below pprint statement
        #pprint.pprint(forest_iris)
        forest_acu_iris = rand_accuracy(test_iris,data_iris,forest_iris)
        print("The random forest accuracy for iris dataset using 2 random features is: ", forest_acu_iris)
        
        forest_iris1 = random_forest(train_iris,data_iris,23,len(train_iris),3,train_iris.columns[:-1].values.tolist())
        forest_acu_iris1 = rand_accuracy(test_iris,data_iris,forest_iris1)
        print("The random forest accuracy for iris dataset using 3 random features is: ", forest_acu_iris1)
    elif x==2:
        train_lens, test_lens = split_dataset(data_lens,8)
        
        tree2 = decision_tree(train_lens,data_lens,train_lens.columns[:-1].values.tolist())
        accur2= accuracy(test_lens,data_lens,tree2)
        print("#######DECISON TREE FOR LENS DATASET#######")
        pprint.pprint(tree2)
        print('The accuracy for len dataset using decision tree is:',accur2)
        
        bagging_lens=random_forest(train_lens,data_lens,5,len(train_lens),4,train_lens.columns[:-1].values.tolist())
        ##3rd argument above is number of bagged trees. You can change the value for testing
        # to see the list of bagged trees remove comment from below pprint statement
        #pprint.pprint(bagging_lens)
        bag_acu_lens=rand_accuracy(test_lens,data_lens,bagging_lens)
        print("The bagging accuracy for the lens dataset is: ", bag_acu_lens)
        
        forest_lens = random_forest(train_lens,data_lens,5,len(train_lens),2,train_lens.columns[:-1].values.tolist())
        #3rd argument is number of trees.You can change the value for testing
        #5th argument is number of features which you can change between 1 to 4 for testing
        #to see the random forest list remove comment from below pprint statement
        #pprint.pprint(forest_lens)
        forest_acu_lens = rand_accuracy(test_lens,data_lens,forest_lens)
        print("The random forest accuracy for lens dataset using 2 random features is: ", forest_acu_lens)
        
        forest_lens1 = random_forest(train_lens,data_lens,5,len(train_lens),3,train_lens.columns[:-1].values.tolist())
        forest_acu_lens1 = rand_accuracy(test_lens,data_lens,forest_lens1)
        print("The random forest accuracy for lens dataset using 3 random features is: ", forest_acu_lens1)
   
    elif x==3:
        train_lens, test_lens = split_dataset(data_lens,8)
        train_iris, test_iris = split_dataset(data_iris,50)
        stump_iris = decision_stump(train_iris,data_iris,train_iris.columns[:-1].values.tolist())
        print("#########DECISION STUMP FOR IRIS DATASET##########")
        print(stump_iris)
        stump_lens = decision_stump(train_lens,data_lens,train_lens.columns[:-1].values.tolist())
        print("#########DECISION STUMP FOR LENS DATASET######")
        print(stump_lens)
    