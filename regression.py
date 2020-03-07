# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 10:27:33 2019

@author: Divya Thakkar
"""


import numpy as np
import pandas as pd
import random
import pprint
data_servo=pd.read_csv("servo.data", sep=',', names=['motor','screw','p_gain','v_gain','class'])
data_crane =pd.read_csv('Container_Crane_Controller_Data_Set.csv',sep=';')
data_crane.replace({'0,3':0.3,'0,5':0.5,'0,7':0.7},inplace=True)

def split_dataset(df,test_size):
    random.seed(2)
    indices=df.index.tolist() # gives list of indices
    test_indices=random.sample(population=indices,k=test_size) #randomply sample k indices
    test_df=df.loc[test_indices] #subset of data(test) with test_indices values
    train_df=df.drop(test_indices) # training data with remaining indices(excluding the test indices)
    
    return train_df,test_df

def variance(data,attr,target):
    feature_values = np.unique(data[attr])
    feature_variance = 0
    var_values=[]
    split_points=[]
    for value in feature_values:
        subset=data.where(data[attr]==value).dropna()
        #Calculate the weighted variance of each subset            
        value_var = (len(subset)/len(data))*np.var(subset[target])
        var_values.append(value_var)
        #print(var_values)
        split_points.append(value)
        #print(split_points)
        idx=np.argmin(var_values)
        best_split_point =split_points[idx]
        #Calculate the weighted variance of the feature
        feature_variance+=value_var
    return feature_variance , best_split_point

def best_feature1(data,feature):
    variance_values=[]
    split_values=[]
    for features in feature:
        split_variance,split_point=variance(data,features,data.columns[-1])
        
        variance_values.append(split_variance)
        
        split_values.append(split_point)
    best_feature_index=np.argmin(variance_values)
    #print(variance_values[best_feature_index])
    best_feature=feature[best_feature_index]
    split_point=split_values[best_feature_index]
    #print('best feature is',best_feature,'with split point',split_point)
    return best_feature,split_point

def type_of_feature(df,feature):
    unique_values = np.unique(df[feature])
    if len(unique_values)<=5 or isinstance(unique_values[0],str):
        feature_type = 'categorical'
    else:
        feature_type = 'continuous'
    return feature_type

def decision_stump(dataset,orignal_dataset,features,min_instances,target_attribute_name):
    if dataset.empty:
        return np.mean(orignal_dataset[target_attribute_name])
    
    if len(dataset) <= int(min_instances):
        return np.mean(dataset[target_attribute_name])
   
    elif len(features) ==0:
        #return default_class
        return np.mean(dataset[target_attribute_name])
    else:
        best_feature, value= best_feature1(dataset,features)
        tree = {best_feature:{}}
        subfeature =[]
        if type_of_feature(orignal_dataset,best_feature)=='continuous':
            df_left=dataset.where(dataset[best_feature]<=value).dropna()#dataframe for left tree
            df_right=dataset.where(dataset[best_feature]>value).dropna()#dataframe for right tree
        
            value_left= '<='+str(value)
            value_right='>'+str(value)
        
            tree[best_feature][value_left]=decision_stump(df_left,orignal_dataset,subfeature,min_instances,target_attribute_name)
            tree[best_feature][value_right]=decision_stump(df_right,orignal_dataset,subfeature,min_instances,target_attribute_name)
        else:
            for value in np.unique(dataset[best_feature]):
            
                subset = dataset.where(dataset[best_feature] == value).dropna() # subtree for every category if the feature is categorical
                tree[best_feature][value] = decision_stump(subset,orignal_dataset,subfeature,min_instances,target_attribute_name)
        return tree
def decision_tree(data,org,features,min_instances,target_attribute_name,feature_randomness=None):
    if data.empty:
        return np.mean(org[target_attribute_name])
    
    if len(data) <= int(min_instances):
        return np.mean(data[target_attribute_name])
   
    elif len(features) ==0:
        #return default_class
        return np.mean(data[target_attribute_name])
    else:
        no_of_features = len(data.columns[:-1].values) #Finding number of features in dataset excluding the label column
        feature_index_list = list(range(no_of_features))
        if feature_randomness and feature_randomness <= no_of_features: 
            feature_indices = random.sample(population=feature_index_list, k=feature_randomness) #randomly selecting subset of feature indices as specified by k to build random forest
            features=[] 
            for feature_index in feature_indices:          
                feat = data.columns[ feature_index]
                features.append(feat)
        #default_class = majority_label(dataset.values)
        best_feature, value= best_feature1(data,features)
        #print("We use "+best_feature+" for splitting")
        tree = {best_feature:{}}
        subfeature =[]
        for i in features:
            if i!=best_feature:
                subfeature.append(i)
            
        if type_of_feature(org,best_feature)=='continuous':
            df_left=data.where(data[best_feature]<=value).dropna()#dataframe for left tree
            df_right=data.where(data[best_feature]>value).dropna()#dataframe for right tree
        #value1=df_left.values
        #value2=df_right.values
            value_left= '<='+str(value)
            value_right='>'+str(value)
        
            tree[best_feature][value_left]=decision_tree(df_left,org,subfeature,min_instances,target_attribute_name)
            tree[best_feature][value_right]=decision_tree(df_right,org,subfeature,min_instances,target_attribute_name)
        else:
            for value in np.unique(data[best_feature]):
                sub_data = data.where(data[best_feature] == value).dropna()
                tree[best_feature][value] = decision_tree(sub_data,org,subfeature,min_instances,target_attribute_name)    
        return(tree)

def pred(orignal_data,Tree,features,test,default_class=None):
    default_class= np.mean(orignal_data[orignal_data.columns[-1]])
    class_Label= default_class
    a = list(Tree.keys())[0]#get the key from dictionary
    b = Tree[a]
    featureindex= features.index(a)
    if type_of_feature(orignal_data,features[featureindex])=='categorical':
        #print('entered categorical loop')
        for key in b.keys():
            if test[featureindex]==key:
                if isinstance(b[key],dict):
                    class_Label = pred(orignal_data,b[key], features, test)
                    #return pred(df,b[key],features,test)
                else:
                    class_Label=b[key]
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
            return pred(orignal_data,ans, features, test)
        else:   
            class_Label=ans
    return class_Label

def error(df,tree,orignal_data):
    #predictionlist = []
    err=0
    for i in range(len(df)):
        ans =pred(orignal_data,tree,orignal_data.columns[:-1].values.tolist(),df.values[i])
        err+= ((ans - df[df.columns[-1]].values[i])**2)
    MSE = np.sqrt(err/len(df)) 
    return MSE

def bootstrap_samples(dataframe,no_of_boot):
    samples=dataframe.sample(n=no_of_boot, replace=True)
    return samples

def random_forest(df,original_dataset,no_of_trees,no_of_boot,no_of_features,features,min_instances,target_attribute):
    forestlist=[]
    for i in range(no_of_trees):
        samples= bootstrap_samples(df,no_of_boot)
        tree =decision_tree(samples,original_dataset,features,min_instances,target_attribute,feature_randomness=no_of_features)
        forestlist.append(tree)
    return forestlist

def random_predict(orignal_data,forest,test_vector):
    predictions=[]
    for i in range(len(forest)):
        predict = pred(orignal_data,forest[i],orignal_data.columns[:-1].values.tolist(),test_vector)
        #print('prediction for tree ' +str(i)+ 'is',predict )
        predictions.append(predict)
    #print(predictions)
    output = np.mean(predictions)
    return output

def rand_error(df,orignal_data,forest):
    err=0
    for i in range(len(df)):
        ans =random_predict(orignal_data,forest,df.values[i])
        #print('actual : '+str(ans)+ ' expected :' ,df[df.columns[-1]].values[i] )
        err+= ((ans - df[df.columns[-1]].values[i])**2)
    answer = np.sqrt(err/len(df))
    #avg=bagmse/len(bag)    
    return answer

if __name__ == '__main__':
    print("1.Implementation of decision trees, bagging and random forests for servo dataset \n2.Implementation of decision trees, bagging and random forests for crane dataset \n3.Implementation of decision stumps for servo and crane dataset ")
    x= int(input('enter choice'))
    if x==1:
        train_servo,test_servo=split_dataset(data_servo,test_size=45)
        #You can change the value of test size from 1 to 165
        tree_servo=decision_tree(train_servo,data_servo,train_servo.columns[:-1].values.tolist(),7,'class')
        #4th argument is minum number of instances. You can change value in proper range for testing
        print("##########DECISION TREE FOR SERVO DATASET#########")
        pprint.pprint(tree_servo)
        error_servo=error(test_servo,tree_servo,data_servo)
        print("The error rate(RMSE) of the decision tree is: ", error_servo)
        
        bagging_servo= random_forest(train_servo,data_servo,50,len(train_servo),4,train_servo.columns[:-1].values.tolist(),7,'class')
        #3rd argument is number of trees which you can change for testing
        #5th argument is number of features which you can change between 1 to 4 for testing random forests
        #7th argument is minimum number of features which you can change for testing. Note minimum number of instances should not be greater than or equal to size of dataset.
        bagerr=rand_error(test_servo,data_servo,bagging_servo)
        print("The error rate(RMSE) for bagging is",bagerr)
        
        forest_servo= random_forest(train_servo,data_servo,40,len(train_servo),2,train_servo.columns[:-1].values.tolist(),7,'class')
        foresterr=rand_error(test_servo,data_servo,forest_servo)
        print("The error rate(RMSE) for random forest with 2 features is",foresterr)
        
        forest_servo1= random_forest(train_servo,data_servo,40,len(train_servo),3,train_servo.columns[:-1].values.tolist(),7,'class')
        foresterr1=rand_error(test_servo,data_servo,forest_servo1)
        print("The error rate(RMSE) for random forest with 3 features is",foresterr1)
        
    if x==2:
        train_crane,test_crane=split_dataset(data_crane,test_size=4)
        #You can change the value of test size from 1 to 12
        tree_crane=decision_tree(train_crane,data_crane,train_crane.columns[:-1].values.tolist(),2,'Power')
        print('########DECISION TREE FOR CONTROLLER CRANE DATASET######')
        pprint.pprint(tree_crane)
        error_crane=error(test_crane,tree_crane,data_crane)
        print('The error rate(RMSE) of decision tree is: ',error_crane)
        
        bagging_crane= random_forest(train_crane,data_crane,10,len(train_crane),3,train_crane.columns[:-1].values.tolist(),2,'Power')
        bagerr=rand_error(test_crane,data_crane,bagging_crane)
        print('The error rate(RMSE) for bagging is :',bagerr)
        
        forest_crane= random_forest(train_crane,data_crane,10,len(train_crane),2,train_crane.columns[:-1].values.tolist(),2,'Power')
        foresterr=rand_error(test_crane,data_crane,forest_crane)
        print('The error rate(RMSE) for random forests using 2 features is: ',foresterr)
        
    if x==3:
        train_servo,test_servo=split_dataset(data_servo,test_size=45)
        train_crane,test_crane=split_dataset(data_crane,test_size=4)
        print('######DECISION STUMPS FOR SERVO DATASET#########')
        stump_servo=decision_stump(train_servo,data_servo,train_servo.columns[:-1].values.tolist(),5,'class')
        pprint.pprint(stump_servo)
        stump_crane=decision_stump(train_crane,data_crane,train_crane.columns[:-1].values.tolist(),2,'Power')
        print('#######DECISION STUMP FOR CRANE DATASET######')
        pprint.pprint(stump_crane)      
        
        
        
        
