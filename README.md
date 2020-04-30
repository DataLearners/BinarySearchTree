# BST

Binary Search Tree is a custom application of the [binary search algorithm](https://www.geeksforgeeks.org/binary-search/) on a given data set. The user must specify which column contains the response or y-variable. Based on the data type of the response variable the module determines whether to utilize Classification Trees or Regression Trees. Thus, mathematically the bst module utilizes the [CART](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/) method. Classification Trees are applied to string responses. Regression Trees are applied to numerical responses.

### Classification Tree

For the classification tree the loss function is the Gini index, where C represents all of the categories in the response variable. p represents the proportion of the response variable belonging to the category. So each node in the tree has a gini score using the formula below.
$$
Gini(node_j) = 1 - \sum_{i=1}^{C} (p_i)^2
$$
Nodes are split in a binary fashion where the categories(C) for each node are all either true or false. The weighted gini calculation for the true-false split on a node follows the formula below. w corresponds to the proportion of the node data that belongs to the category.
$$
Gini_w(node_j) = w_t*g(C_t) + w_f * g(C_f)
$$
The information gained by the node is the following formula. The algorithm searches for the split that results in the highest information gain. w corresponds to the proportion of the total data that exists in the node relative to the entire data set.
$$
Gain = w_{node} * (Gini(node) - Gini_w(node))
$$

### Regression Tree

For the regression tree the loss function is Variance, so each node in the tree has an impurity score using the formula below.
$$
Var(node_j) = \sum_{i=1}^{n} \frac{(x_i - \bar{X})}{n-1}
$$
The weighted and gain functions follow the same structure as the Classification tree. Thus, the formulas are:
$$
Var_w(node_j) = w_t*var(C_t) + w_f * var(C_f)
$$

$$
Gain = w_{node} * (var(node) - var_w(node))
$$

### Random Forest

The random forest is a collection of either Classification Trees or Regression Trees. The Random Forest uses an ensemble technique to make predictions. In the case of a Classification Forest the prediction is the mode of all Classification Trees. In the case of a Regression Forest, the prediction is the mean of all Regression Trees.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install bst.

```
cmd> cd 'C:\...\dist'
cmd> python -m pip install bst-1.0.2-py3-none-any.whl
```



## Usage

```python
import bst
# Sample data of fruit with diameters and colors
import random
size = 10
diameters = [random.randint(1, 6) for i in range(size)]
fruits = random.choices(['Apple', 'Grape', 'Lemon'], k=size)
colors = random.choices(['Green', 'Yellow', 'Red'], k=size)
for i, fruit in enumerate(fruits):
    if fruit == 'Lemon':
        colors[i] = 'Yellow'
        diameters[i] = random.triangular(3, 4, 5)
    if fruit == 'Grape':
        colors[i] = 'Green'
        diameters[i] = random.triangular(1, 1.5, 2)
    if fruit == 'Apple':
        colors[i] = random.choice(['Red', 'Green'])
        diameters[i] = random.triangular(3, 5, 6)
header = ["color", "diameter", "fruit"]
data = [list(row) for row in zip(colors, diameters, fruits)]

"""To build an individual tree, utilize the Model Data Frame class. Use the Model Data 
Frame class to specify which column contains the response variable. Once the response 
variable is specified, the algorithm automatically decides whether to build a 
Classification Tree or Regression Tree. The tree parameters control the growth 
of the tree. """
# Classification Tree
fruitdf = bst.ModelDF(bst.ListDF(data, header), ycol=0) 
ginitree = bst.Tree(fruitdf, mingain=0, minrows=1, maxheight=10)
# Regression Tree
fruitdf = bst.ModelDF(bst.ListDF(data, header), ycol=1)
regtree = bst.Tree(fruitdf, conf=0.95)

"""The Forest class adds additional parameters to control the growth of trees 
within the forest. Since Random Forests use a subset of the available features 
to find the best splits, the user will set the number of features a priori."""
# Classification Forest
fruitdf = bst.ModelDF(bst.ListDF(data, header), ycol=0)
ginifrst = bst.Forest(fruitdf, mingain=0, minrows=1, nfeatures=1, ntrees=25)
# Regression Forest
fruitdf = bst.ModelDF(bst.ListDF(data, header), ycol=1)
regfrst = bst.Forest(fruitdf, conf=0.9, nfeatures=1, ntrees=20)
```
