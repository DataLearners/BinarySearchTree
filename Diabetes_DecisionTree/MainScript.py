# -*- coding: utf-8 -*-
"""
Split dataset into training set and test set
70% training and 30% test
Key attributes of Tree are leaf_matrix, marked_data, tree_array, predict()
"""
import random, matplotlib.pyplot as plt, numpy as np
import startup, prep, bst
import prep.ingest as ingest

random.seed(10)
datapath = 'data_science_poc\\Diabetes_DecisionTree\\Dataset'
datafolder = startup.find_subroot(datapath)
startup.make_folder('Output')
outputpath = startup.find_subroot('Diabetes_DecisionTree\\Output')
files = ingest.Folder(datafolder)
files()
 
yes_no = lambda x: {1:'Yes', 0:'No'}[x]
df = files.toListDF(fileindex=0)
col = [yes_no(random.randint(0, 1)) for i in range(df.num_rows)]
df.add_col(col, 'HeartAttack')
df.transform_col(yes_no, 'Outcome')
train_data, test_data = prep.split_train_test(df.data, tst_prop=0.3)

y = df.header.index('Outcome')
diab_tree = bst.Tree(train_data, y, df.header, loss='gini', n_features=9)
diab_pred = diab_tree.predict(test_data)
leaf = random.sample(diab_tree.leaves, 1)[0]
diab_tree.traverse_to(leaf)
diab_tree.leaf_matrix.export("Diabetes_LeafMatrix.csv", folder=outputpath)
diab_tree.tree_array.export("Diabetes_Tree.csv", folder=outputpath)
filename= "Diabetes_MarkedTrainingData.csv"
diab_tree.marked_data.export(filename, folder=outputpath)

y = df.header.index('BMI')
bmi_tree = bst.Tree(train_data, y, df.header, loss='var', 
                    min_rows=15, n_features=9)
filename = "BMI_Predictions.csv"
bmi_tree.predict(test_data, conf=0.8).export(filename, folder=outputpath)
leaf = random.sample(bmi_tree.leaves, 1)[0]
bmi_tree.traverse_to(leaf)
bmi_tree.leaf_matrix.export("BMI_LeafMatrix.csv", folder=outputpath)
bmi_tree.tree_array.export("BMI_Tree.csv", folder=outputpath)
bmi_tree.marked_data.export("BMI_MarkedTrainingData.csv", folder=outputpath)

rows = 0
horiz, mse, rsq, ftrs = [], [], [], []
y = df.header.index('BMI')
for i in range(20):
    horiz.append(rows)
    rows += 5
    bmi_tree = bst.Tree(train_data, y, df.header, loss='var', 
                        min_rows=rows, n_features=9)
    print("Min Rows {}".format(rows))
    pred = bmi_tree.predict(test_data, conf=0.9)
    mse.append(bmi_tree.mse)
    rsq.append(bmi_tree.r_sq)
    ftrs.append(len(bmi_tree.feature_importance))
  
plt.figure(figsize=(18, 6))
plt.subplot(1,3,1)    
x = np.asarray(horiz)
plt.plot(x, np.asarray(mse))
plt.axvline(x=50, ls=':')
plt.ylabel('MSE')
plt.xlabel('Minimum Node Size')
plt.subplot(1,3,2)    
x = np.asarray(horiz)
plt.plot(x, np.asarray(rsq))
plt.axvline(x=50, ls=':')
plt.ylabel('RSquare')
plt.xlabel('Minimum Node Size')
plt.subplot(1,3,3)    
x = np.asarray(horiz)
plt.plot(x, np.asarray(ftrs))
plt.axvline(x=50, ls=':')
plt.ylabel('Features')
plt.xlabel('Minimum Node Size')
plt.savefig('Output\\AccuracyPlots.png')
plt.show()

bst.config.TIMES.plot_times()

