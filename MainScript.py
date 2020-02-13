# -*- coding: utf-8 -*-
import startup
startup.add_library('DataLearners\\RandomForest\\library')

import random, matplotlib.pyplot as plt, numpy as np
import bst
"""
Split dataset into training set and test set
70% training and 30% test
Key attributes of Tree are leaf_matrix, marked_data, tree_array, predict()
"""
random.seed(10)
datapath = 'DataLearners\\RandomForest\\Dataset'
datafolder = startup.find_subroot(datapath)
startup.make_folder('Output')
outputpath = startup.find_subroot('DataLearners\\RandomForest\\Output')
files = bst.Folder(datafolder)
files()
 
yes_no = lambda x: {1:'Yes', 0:'No'}[x]
df = files.to_list_df(fileindex=0)
col = [yes_no(random.randint(0, 1)) for i in range(df.num_rows)]
df.add_col(col, 'HeartAttack')
df.transform_col(yes_no, 'Outcome')

header = ['Response', 'X1', 'X2']
tri = lambda low, high, mode: random.triangular(low, high, mode)
y = [random.random() for i in range(1000)]
data = [[y[i], tri(0.4, 0.6, 0.5)*y[i], tri(0.3, 0.7, 0.5)*y[i]] for i in range(1000)]
df = bst.ListDF(data, header)

train_data, test_data = bst.split_train_test(df.data, tst_prop=0.3)

##############Decision Tree#########################################
y = df.header.index('Outcome')
diab_tree = bst.Tree(train_data, y, df.header, lossfunc='gini')
diab_pred = bst.Predict(diab_tree, test_data).df
leaf = random.sample(diab_tree.leaves, 1)[0]
diagnostics = bst.Summary(diab_tree)
diagnostics.traverse_to(nodeidx=leaf)
diagnostics.leafmatrix.export("Diabetes_LeafMatrix.csv", folder=outputpath)
diagnostics.nodesummary.export("Diabetes_Tree.csv", folder=outputpath)
filename= "Diabetes_MarkedTrainingData.csv"
diagnostics.leafclusters.export(filename, folder=outputpath)

y = df.header.index('BMI')
bmi_tree = bst.Tree(train_data, y, df.header, lossfunc='var', min_rows=15)
filename = "BMI_Predictions.csv"
bst.Predict(bmi_tree, test_data, conf=0.8).df.export(filename, folder=outputpath)
leaf = random.sample(bmi_tree.leaves, 1)[0]
diagnostics = bst.Summary(bmi_tree)
diagnostics.traverse_to(nodeidx=leaf)
diagnostics.leafmatrix.export("BMI_LeafMatrix.csv", folder=outputpath)
diagnostics.nodesummary.export("BMI_Tree.csv", folder=outputpath)
diagnostics.leafclusters.export("BMI_MarkedTrainingData.csv", folder=outputpath)

##########Random Forest#####################################
y = df.header.index('Outcome')
diab_forest = bst.Forest(train_data, y, df.header, lossfunc='gini', n_features=2)
print("Outcome Out of bag error {}".format(diab_forest.oob_err))
diab_pred = bst.Predict(diab_forest, test_data).df

y = df.header.index('BMI')
bmi_forest = bst.Forest(train_data, y, df.header, lossfunc='var', n_features=3)
print("BMI Out of bag error {}".format(bmi_forest.oob_err))
diab_pred = bst.Predict(diab_forest, test_data, conf=0.9).df
bst.Summary(bmi_forest).nodesummary.export("BMI_ForestLeafMatrix.csv", folder=outputpath)
bst.Summary(bmi_forest).leafclusters.export("BMI_ForestMarkedTrainingData.csv", folder=outputpath)

horiz = [trees for trees in range(3, 9, 3)]
rsq, obe, mse = [], [], []
for trees in horiz:
    reg_forest = bst.Forest(train_data, 5, df.header, lossfunc='var', 
                            n_features=2, n_trees=trees)
    print("{} trees Out bag error {}".format(trees, reg_forest.oob_err))
    reg_pred = bst.Predict(reg_forest, test_data)
    rsq.append(reg_pred.rsq)
    obe.append(reg_forest.oob_err)
    mse.append(reg_pred.mse) 
  
plt.figure(figsize=(15, 12))
plt.subplot(2, 2, 1)    
x = np.asarray(horiz)
plt.plot(x, np.asarray(rsq))
plt.ylabel('R-Square')
plt.xlabel('Trees')
plt.subplot(2, 2, 2) 
plt.plot(x, np.asarray(mse), label='MeanSqError')
plt.ylabel('MeanSqError')
plt.xlabel('Trees')
plt.subplot(2, 2, 3)
plt.plot(x, np.asarray(obe), label='out of bag error')
plt.ylabel('Out of Bag Error')
plt.xlabel('Trees')
plt.savefig('Output\\AccuracyPlots.png')
plt.show()


