# -*- coding: utf-8 -*-
"""
Split dataset into training set and test set
70% training and 30% test
Key attributes of Tree are leaf_matrix, marked_data, tree_array, predict()
"""
import random, startup, prep, bst
import prep.ingest as ingest
from sklearn import metrics

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
 
bmi_tree = bst.Regr(train_data, df.header.index('BMI'), df.header,
                          min_rows=2, confidence=0.9)
bmi_pred = bmi_tree.predict(test_data)
val = metrics.mean_squared_error(bmi_pred.BMI, bmi_pred.Predicted)

print("Mean Square Error {}".format(val))
