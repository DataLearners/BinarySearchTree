# -*- coding: utf-8 -*-
"""
Split dataset into training set and test set
70% training and 30% test
Key attributes of Tree are leaf_matrix, marked_data, tree_array, predict()
"""
import startup, random, prep, bst, matplotlib.pyplot as plt, numpy as np
import prep.ingest as ingest
from forest import Forest

random.seed(10)
datafolder = startup.find_subroot('Python\\Diabetes_DecisionTree\\Dataset')
files = ingest.Folder(datafolder)
files()
 
yes_no = lambda x: {1:'Yes', 0:'No'}[x]
df = files.toListDF(fileindex=0)
col = [yes_no(random.randint(0, 1)) for i in range(df.num_rows)]
df.add_col(col, 'HeartAttack')
df.transform_col(yes_no, 'Outcome')
train_data, test_data = prep.split_train_test(df.data, tst_prop=0.3)

resp = df.header.index('Outcome')
diab_forest = Forest(train_data, resp, df.header, loss='gini', n_features=2)
print("Outcome Out of bag error {}".format(diab_forest.out_bag_error))
a = diab_forest.predict(test_data)

resp = df.header.index('BMI')
bmi_forest = Forest(train_data, resp, df.header, loss='var', n_features=3)
print("BMI Out of bag error {}".format(bmi_forest.out_bag_error))
b = bmi_forest.predict(test_data, conf=0.85)

trees = 3
horiz, rsq, obe, mse = [], [], [], []
for i in range(5):
    horiz.append(trees)
    reg_forest = Forest(train_data, 5, df.header, loss='var',
                        n_features=2, n_trees=trees)
    print("{} trees Out of bag error {}".format(trees, reg_forest.out_bag_error))
    pred = reg_forest.predict(test_data)
    rsq.append(reg_forest.r_sq)
    obe.append(reg_forest.out_bag_error)
    mse.append(reg_forest.mse)
    trees += 3
  
plt.figure(figsize=(18, 6))
plt.subplot(1,2,1)    
x = np.asarray(horiz)
plt.plot(x, np.asarray(rsq))
plt.ylabel('R-Square')
plt.xlabel('Trees')
plt.subplot(1,2,2) 
plt.plot(x, np.asarray(mse), label='MeanSqError')
plt.plot(x, np.asarray(obe), label='out of bag error')
plt.ylabel('Error')
plt.xlabel('Trees')
plt.legend()
plt.savefig('AccuracyPlots.png')
plt.show()

bst.config.TIMES.plot_times()