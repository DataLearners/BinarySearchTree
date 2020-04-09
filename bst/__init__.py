"""The Binary Search Tree module utilizes tree and forest hyperparameters
to construct binary search trees from Nodes, Questions, and Splits. The
binary search is a recursive greedy algorithm"""

import os
import csv
import copy
import math
import numbers
import statistics
import random
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats
import pprint as pp
import time
start = time.time()


def flatten(nested_list):
    """Flattens a nested list into a single list"""
    def flatten_generator(nested_list):
        for i in nested_list:
            if isinstance(i, (list, tuple)):
                for j in flatten_generator(i):
                    yield j
            else:
                yield i
    return list(flatten_generator(nested_list))


def neat_string(matrix, sigdigits=3):
    """Neat representation of data"""
    cap = lambda x: round(x, sigdigits) if isinstance(x, float) else x
    digit = [[cap(x) for x in row] for row in matrix]
    strng = [[str(e) for e in row] for row in digit]
    lens = [max(map(len, col)) for col in zip(*strng)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in strng]
    return '\n'.join(table)


# #####################Settings###############################
class Loss:
    """Controls the types of Decision Trees generated based on the dataset.
    Data with a categorical response will utilize the gini loss function and
    produce Classification Trees. Data with a numerical response will
    utilize the variance loss function and produce Regression Trees."""

    def __init__(self, modeldf):
        self.name = None
        self.func = None
        self.predict = None
        self.ycol = modeldf.ycol
        self.set_lossfunc(modeldf(modeldf.ycol))

    def mode(self, x):
        """Custom mode function"""
        if len(x) > 0:
            return max(set(x), key=x.count)
        return ''

    def set_lossfunc(self, ydata):
        """Sets the method for making predictions and calculating impurity"""
        if any([isinstance(item, str) for item in ydata]):
            self.name = 'gini'
            self.func = self.gini
            self.predict = self.mode
            return
        if all([isinstance(item, numbers.Number) for item in ydata]):
            self.name = 'var'
            self.func = self.var
            self.predict = scipy.mean
            return

    def class_counts(self, data):
        """Counts the number of each category."""
        counts = {}
        for row in data:
            label = row[self.ycol]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def gini(self, data):
        """Calculate the Gini Impurity for a list of rows"""
        counts = self.class_counts(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / len(data)
            impurity -= prob_of_lbl**2
        return impurity

    def var(self, data):
        """Calculate variance of the response column"""
        response = [row[self.ycol] for row in data]
        if len(response) > 1:
            return statistics.variance(response)
        return 0


class TreeParams:
    """Hyperparameters for controlling growth of trees and forest.
    Tree Settings
    Mingain--sets the minimum information gain that must occur for splitting.
    Minrows--sets the minimum number of rows that must be in a Leaf Node.
    Maxhgt--sets the maximum height for a tree"""
    def __init__(self, mingain, minrows, maxheight):
        self.mingain = mingain
        self.minrows = minrows
        self.maxheight = maxheight

    def gain_enough(self, split):
        """Determine if the split has gained sufficient information"""
        return split.info_gain >= self.mingain

    def big_enough(self, split):
        """Determines if the split has sufficient size"""
        sizes = map(len, [split.true_rows, split.false_rows])
        size = min(sizes, default=0)
        return size >= self.minrows

    def short_enough(self, split):
        """Determine if the split exceeds the height preset for the tree"""
        height = round(math.log2(split.index + 1))
        return height < self.maxheight

    def isvalid(self, split):
        """Determine if the split should be allowed given the settings"""
        enough = [self.gain_enough(split), self.big_enough(split),
                  self.short_enough(split)]
        return all(enough)


class Settings:
    """Nfeatures--controls number of features selected on Tree in Forest
    Ntrees--the number of trees in the forest
    """
    def __init__(self, modeldf, tparams, ntrees=1, nfeatures=2):
        self.modeldf = modeldf
        self.tparams = tparams
        self.ntrees = ntrees
        self.nfeatures = self.set_nfeatures(ntrees, nfeatures)
        self.questions = {}
        self.loss = Loss(modeldf)

    def set_nfeatures(self, ntrees, nfeatures):
        """Sampling a subset of the features only applies to Forests. Trees
        always utilize all the features for determining a best split"""
        if ntrees > 1:
            return nfeatures
        return self.modeldf.xdf.numcols

    def feature_sample(self, featurecols):
        """Random features sample for determining best split in Forest"""
        return random.sample(featurecols, self.nfeatures)


# #######################Tree Parts########################
class Question:
    """Question is used to partition a dataset into true and false answers"""
    def __init__(self, col_num, value, header):
        self.header = header
        self.col_num = col_num
        self.value = value
        self.col_name = header[col_num]
        self.numeric = isinstance(value, numbers.Number)

    def match(self, row):
        """Compare the feature value in a row to the
        feature value in question."""
        val = row[self.col_num]
        if self.numeric:
            return val >= self.value
        return val == self.value

    def __repr__(self):
        condition = "=="
        val = self.value
        if self.numeric:
            val = round(self.value, 3)
            condition = ">="
        str_val = str(val)
        return "Is {} {} {}?".format(self.col_name, condition, str_val)


class Node:
    """Generic object attached to a binary search tree. Object converts into
    a Leaf or Decision depending on the best split possible."""
    def __init__(self, data, settings, idx):
        self.index = idx
        self.data = data
        self.score = settings.loss.func(data)
        self.gain = None
        self.question = None
        self.settings = settings
        self.bestsplit(data)

    def bestsplit(self, data):
        """Find the best question to ask by iterating over every feature
        and calculating the information gain."""
        features = self.settings.feature_sample(self.settings.modeldf.xcols)
        header = self.settings.modeldf.header
        col_qns = [Column(data, fcol, header).questions for fcol in features]
        questions = flatten(col_qns)  # unpack sublists for each column
        self.question, self.gain = None, 0  # default values
        isgood = lambda x: all([x.valid, x.info_gain > self.gain])
        for question in questions:
            split = Split(self, question)
            if isgood(split):
                self.question, self.gain = question, split.info_gain


class Leaf:
    """A Leaf node predicts the response based on the data. It predicts
    mode in the case of Classification Tree and mean in the case of
    Regression Tree"""
    def __init__(self, node):
        [setattr(self, attr, val) for attr, val in node.__dict__.items()]
        self.branch = self.path(node.index)
        self.ytrue = [row[node.settings.modeldf.ycol] for row in node.data]
        self.prediction = self.get_prediction(node)
        self.treeid = Tree.counter
        self.numrows = len(node.data)
        self.type = type(self).__name__
        self.features = self.indicators(node)

    def get_prediction(self, node):
        pred_func = node.settings.loss.predict
        return pred_func(self.ytrue)

    def path(self, idx, ancestors=None):
        """Recursive function to find the ancestors of a particular node
        ancestors = None is necessary to keep the recursive function from
        storing ancestors between function calls which resuts in duplicates
        """
        if ancestors is None:
            ancestors = []
        parent = math.floor(idx/2)
        if parent == 0:
            ancestors.append(idx)
            return sorted(ancestors)
        ancestors.append(idx)  # Due to recursion its idx and not parent
        return self.path(parent, ancestors)

    def indicators(self, node):
        """Recall the features used to develop the leaf"""
        leafpath = self.path(node.index)[:-1]  # ancestor nodes
        qdict = node.settings.questions
        fcols = [qdict[k].col_num for k in leafpath if k in qdict]  # features
        numfeatures = node.settings.modeldf.numcols
        return [1 if x in fcols else 0 for x in range(numfeatures)]


class Decision:
    """A Decision Node asks a question. The question results in a true branch
    and a false branch in response to the question."""
    def __init__(self, left, right, node):
        [setattr(self, attr, val) for attr, val in node.__dict__.items()]
        self.true_branch = left
        self.false_branch = right
        self.treeid = Tree.counter
        self.numrows = len(node.data)
        self.type = type(self).__name__


class Split:
    """Split rows at a specific value of a feature"""
    def __init__(self, node, question):
        self.index = node.index  # required for Tree search settings
        self.question = question
        self.weight = len(node.data)/node.settings.modeldf.numrows
        self.true_rows, self.false_rows = self.partition(node, question)
        self.wgtd_score = self.calc(self.true_rows, self.false_rows, node)
        self.info_gain = self.weight*(node.score - self.wgtd_score)
        self.valid = node.settings.tparams.isvalid(self)

    def partition(self, node, question):
        """Partition dataset by answer to the class: question into
        true rows and false rows"""
        true_rows = [row for row in node.data if question.match(row)]
        false_rows = [row for row in node.data if not question.match(row)]
        return true_rows, false_rows

    def calc(self, left, right, node):
        """Weighted impurity based on size of the child nodes"""
        prop_left = len(left) / (len(left) + len(right))
        impurity_left = node.settings.loss.func(left)
        impurity_right = node.settings.loss.func(right)
        return prop_left * impurity_left + (1 - prop_left) * impurity_right


class Column:
    """Data class for a feature column to store all the attributes used from
    a column in building a binary search tree"""
    def __init__(self, rows, colidx, header):
        self.data = [row[colidx] for row in rows]
        self.uniques = self.find_unique(rows, colidx)
        self.numeric = self.is_numeric(self.uniques)
        self.categorical = any([type(elm) == str for elm in self.uniques])
        self.breakpts = self.gen_breakpts()
        self.questions = [Question(colidx, pt, header) for pt in self.breakpts]

    def is_numeric(self, coldata):
        """Test if column values are numeric"""
        return all([isinstance(value, numbers.Number) for value in coldata])

    def find_unique(self, rows, colidx):
        """Find the unique values for a column in a dataset."""
        vals = set([row[colidx] for row in rows])
        return sorted(list(vals))

    def gen_breakpts(self):
        """Find all the candidate values for splitting the data on a
        given feature into groups. Return k-1 candidates for string variables.
        """
        if self.categorical:
            return self.uniques[1:]
        cuts = range(0, len(self.uniques), 2)
        pairs = [self.uniques[i:i + 2] for i in cuts]
        return [scipy.mean(group) for group in pairs]

    def get_questionlist(self, col, header):
        """All questions for the column object"""
        return [Question(col, pt, header) for pt in self.breakpts]


# #######################Diagnostics########################
class Diagnostics:
    """Generates reports on Decision Tree object for training data frame"""
    def __init__(self, nodes, featnames):
        leafnodes = [node for node in nodes if isinstance(node, Leaf)]
        decisions = [node for node in nodes if isinstance(node, Decision)]
        self.leafclusters = self.get_clusters(leafnodes, featnames)
        self.nodesummary = self.summarize(nodes)
        self.featureimportance = self.scorefeatures(decisions, featnames)
        self.leafmatrix = self.get_matrix(leafnodes, featnames)

    def getattributes(self, nodes, attributes):
        """Extract the attributes of each node into a list with a default
        value for missing values"""
        get = lambda node, attrbs: [getattr(node, x, None) for x in attrbs]
        return [flatten(get(node, attributes)) for node in nodes]

    def get_matrix(self, nodes, featnames):
        """Shows all the features used to make each leaf and the amount of data
        used by the leaf"""
        attributes = ['treeid', 'index', 'prediction', 'numrows', 'features']
        data = self.getattributes(nodes, attributes)
        header = ['treeid', 'index',  'prediction', 'numrows'] + featnames
        return ListDF(data, [x.title() for x in header])

    def get_clusters(self, nodes, featnames):
        """Custom function to generate data clusters associated with leaves"""
        cluster = lambda leaf, row: flatten([leaf.treeid, leaf.index, row])
        data = [cluster(leaf, row) for leaf in nodes for row in leaf.data]
        header = ['treeid', 'index'] + featnames
        return ListDF(data, [x.title() for x in header])

    def summarize(self, nodes):
        """Summarize all of the data contained in each node"""
        attributes = ['treeid', 'index', 'type', 'question', 'prediction']
        attributes = attributes + ['numrows', 'score', 'gain']
        data = self.getattributes(nodes, attributes)
        df = ListDF(data, [x.title() for x in attributes])
        df.transform_col(str, 'Index')
        return df

    def tallygain(self, nodes, featnames):
        """Sum the information gained from every decision node"""
        gain = {x: 0 for x in featnames}
        for node in nodes:
            feature = node.question.col_name
            gain[feature] += node.gain
        sortedkeys = sorted(gain, key=gain.get, reverse=True)
        return {key: gain[key] for key in sortedkeys}

    def scorefeatures(self, nodes, featnames, standardize=True):
        """Determine how much each feature contributes to the model"""
        gain = self.tallygain(nodes, featnames)
        if standardize:
            total = sum([value for value in gain.values()])
            if total > 0:
                gain = {k: gain[k]/total for k in gain}
        return gain


# ########################Predict##################################
class GOF:
    """Method class to contain goodness of fit calculation measures"""
    def __init__(self, ytrue, yhat, leafnodes, settings, conf):
        self.error = self.calc_error(settings.loss, ytrue, yhat)
        self.predictors = len(settings.modeldf.xcols)
        self.intervals = None
        self.rsq = None
        self.adjrsq = None
        self.regression_gof(settings, ytrue, yhat, leafnodes, conf)

    def calc_error(self, loss, ytrue, yhat):
        """Error is measured as match accuracy for gini and mean square error
        for variance"""
        if loss.name == 'gini':
            accuracy = [y == yhat[i] for i, y in enumerate(ytrue)]
            return accuracy.count(False)/len(ytrue)
        return mean_squared_error(ytrue, yhat)

    def regression_gof(self, settings, ytrue, yhat, leafnodes, conf):
        if settings.loss.name == 'var':
            self.intervals = self.calc_intervals(leafnodes, conf)
            self.rsq = r2_score(ytrue, yhat)
            self.adjrsq = self.calc_adjrsq(len(ytrue), self.predictors)

    def conf_int(self, ylist, conf):
        """For the variance loss function, return the mean of the y feature
        and a confidence interval around the mean."""
        mu = scipy.mean(ylist)
        sd = self.calc_stddev(ylist)
        if sd > 0:
            return scipy.stats.norm.interval(conf, mu, sd)
        return mu, mu

    def calc_stddev(self, ylist):
        """Scipy throws an error if the list does not have more than one
        observation"""
        if len(ylist) > 1:
            return scipy.stats.sem(ylist)
        return 0

    def calc_intervals(self, leafnodes, conf):
        """Calculate the confidence intervals on each leaf or set of leaves
        in the case of Forest"""
        # flatten to collapse forest nodes into one list
        ygroups = [flatten([node.ytrue for node in row]) for row in leafnodes]
        return [self.conf_int(y, conf) for y in ygroups]

    def calc_adjrsq(self, numrows, predictors):
        """Proportion of variation in the dependent variable explained by
        the independent variables"""
        denominator = numrows - predictors - 1
        numerator = (1 - self.rsq) * (numrows - 1)
        if denominator > 0:
            ratio = numerator/denominator
            return 1 - ratio

class Fit:
    """Store attributes related to goodness of fit and prediction results"""
    def __init__(self, obj, data):
        self.leafnodes = self.get_leafnodes(obj, data)
        self.yhat = self.calc_yhat(obj, data)
        self.ytrue = self.get_ytrue(obj, data)
        xcols = obj.configs.modeldf.xcols
        self.xmatrix = [[row[i] for i in xcols] for row in data]
        self.xnames = [obj.configs.modeldf.header[i] for i in xcols]
        self.settings = obj.configs

    def get_leafnodes(self, obj, data):
        if isinstance(obj, Tree):
            trees = {0: obj}.values()
        elif isinstance(obj, Forest):
            trees = obj.trees.values()
        return [[tree.classify(row) for tree in trees] for row in data]

    def get_ytrue(self, obj, data):
        """Extract response variable from data"""
        settings = obj.configs
        ycolnum = settings.modeldf.ycol
        return [row[ycolnum] for row in data]

    def calc_yhat(self, obj, data):
        """Calculate the predictions for each row"""
        leafnodes = self.get_leafnodes(obj, data)
        settings = obj.configs
        calcs = [[node.prediction for node in row] for row in leafnodes]
        return [settings.loss.predict(row) for row in calcs]


class Predict:
    """Generates predictions from Decision Tree or Random Forest object"""
    def __init__(self, obj, conf):
        train = obj.configs.modeldf.train
        test = obj.configs.modeldf.test
        self.training = self.gen_predictions(obj, train, conf)
        self.testing = self.gen_predictions(obj, test, conf)
        self.fitsummary = self.get_summary(obj, conf, train, test)

    def gen_predictions(self, obj, data, conf):
        """Make dataframe of prediction results"""
        fit = Fit(obj, data)
        pred_data = [flatten(x) for x in zip(fit.yhat, fit.ytrue, fit.xmatrix)]
        gof = GOF(fit.ytrue, fit.yhat, fit.leafnodes, fit.settings, conf)
        header = flatten(['yhat', 'ytrue', fit.xnames])
        if gof.intervals is not None:
            [x.insert(1, gof.intervals[i]) for i, x in enumerate(pred_data)]
            header.insert(1, 'Intervals')
        return ListDF(pred_data, header)

    def gof_dict(self, obj, data, conf):
        """Goodness of fit dictionary for a particular dataset"""
        fit = Fit(obj, data)
        gof = GOF(fit.ytrue, fit.yhat, fit.leafnodes, fit.settings, conf)
        return gof.__dict__

    def get_summary(self, obj, conf, train, test):
        """Create the goodness of fit data summary"""
        train = self.gof_dict(obj, train, conf)
        test = self.gof_dict(obj, test, conf)
        metrics = ['error', 'predictors', 'rsq', 'adjrsq']
        data = [[k, train[k], test[k]] for k in metrics]
        header = ['Metrics', 'Training', 'Testing']
        return ListDF(data, header)


# ############################Tree#####################################
class Tree(Diagnostics, Predict):
    """Decision Tree class.  Classification Trees and Regression Trees
    depend on the loss function type."""
    counter = 0
    t0 = time.time()

    def __init__(self, modeldf, mingain=0, minrows=1, maxheight=100,
                 ntrees=1, nfeatures=1, conf=0.95):
        Tree.counter += 1
        self.treeid = Tree.counter
        tparams = TreeParams(mingain, minrows, maxheight)
        self.configs = Settings(modeldf, tparams, ntrees, nfeatures)
        self.model = self.build_tree(modeldf.train, self.configs)
        self.nodes = self.flatten_tree()
        nodes = list(self.nodes.values())
        self.height = self.calc_height(nodes)
        Diagnostics.__init__(self, nodes, modeldf.header)
        Predict.__init__(self, self, conf)

    def build_tree(self, data, settings, idx=1):
        """Builds the tree.
        Base case: no further info gain. Since we can ask no further
        questions, we'll return a leaf.
        Otherwise: Continue recursively down both branches. Return a Decision
        node. The Decision node records the question and both branches. """
        node = Node(data, settings, idx)
        t0 = Tree.t0
        t1 = time.time()
        treeidx = Tree.counter
        print("Tree {} Node {} Completed at {}".format(treeidx, idx, t1-t0))
        if node.question is None:
            return Leaf(node)

        split = Split(node, node.question)
        settings.questions[idx] = node.question
        left = self.build_tree(split.true_rows, settings, 2*idx)
        right = self.build_tree(split.false_rows, settings, 2*idx + 1)
        return Decision(left, right, node)

    def flatten_tree(self):
        """Macro function necessary to prevent nodes from migrating to other
        trees in a forest"""
        def attach_nodes(node, treenodes=None):
            """Add Leaf and Decision objects to the tree"""
            if treenodes is None:
                treenodes = {}
            if isinstance(node, Leaf):
                treenodes[node.index] = node
            if isinstance(node, Decision):
                treenodes[node.index] = node
                attach_nodes(node.true_branch, treenodes)
                attach_nodes(node.false_branch, treenodes)
            return {k: treenodes[k] for k in sorted(treenodes)}
        return attach_nodes(self.model)

    def print_(self):
        """Builds a list of strings"""
        def tree_string(node, tree_repr=None):
            """print_tree(my_tree.model)"""
            if tree_repr is None:
                tree_repr = []
            if isinstance(node, Leaf):
                tree_repr.append("-->{}".format(node.prediction))

            if isinstance(node, Decision):
                tree_repr.append("\n{}".format(node.question))

                tree_repr.append("\n\tTrue")
                tree_string(node.true_branch, tree_repr)
                tree_repr.append("\tFalse")
                tree_string(node.false_branch, tree_repr)
            return tree_repr
        return "".join(tree_string(self.model))

    def classify(self, row):
        """Macro function of get_leaf to return leaf nodes from tree based
        on data. Makes application simpler in other classes."""
        def get_leaf(row, node):
            """Recursive function for returning Leaf node from tree.model
            Base case: reach a Leaf node
            Recursion: traverse Decision Nodes along branches until Base Case
            classify(row, self.model)
            """
            if isinstance(node, Leaf):
                return node
            if node.question.match(row):
                return get_leaf(row, node.true_branch)
            return get_leaf(row, node.false_branch)
        return get_leaf(row, self.model)

    def calc_height(self, nodes):
        """Size of a binary tree of height, h is 2^h - 1
        Height = log2(n+1) where n is the max node"""
        idx = max([node.index for node in nodes], default=1)
        h = round(math.log2(idx + 1))
        return h

    def __call__(self, index=1):
        items = self.nodes[index].__dict__.items()
        ugly = ['data', 'settings', 'false_branch', 'true_branch']
        pp.pprint({k: v for k, v in items if k not in ugly})
        return self.nodes[index]


# ####################Forest############################################
class OutofBag:
    """Calculation of out of bag error for Random Forest. The number of spills
    is the same as the number of trees in the forest. The length of yhat and
    ytrue should also be the same as the number of trees."""
    def __init__(self, forest, conf):
        self.aggregator = forest.configs.loss.predict
        self.ytrue = self.aggregate_y(forest)
        self.leaves = self.get_leaves(forest)
        self.yhat = self.predict()
        args = [self.ytrue, self.yhat, self.leaves, forest.configs, conf]
        self.error = GOF(*args).error

    def aggregate_y(self, forest):
        """Get yvalues from all the spills"""
        spills = forest.spills.values()
        ycol = forest.configs.modeldf.ycol
        ytrue = lambda x: self.aggregator([row[ycol] for row in x])
        return flatten([ytrue(spill) for spill in spills])

    def oobleaves(self, row, forest):
        """Get leaf nodes not built with the row of data"""
        oobkeys = [k for k, boot in forest.boots.items() if row not in boot]
        subforest = [v for k, v in forest.trees.items() if k in oobkeys]
        return [tree.classify(row) for tree in subforest]

    def get_leaves(self, forest):
        """Return the subforest that is out of the bag"""
        spills = forest.spills.values()
        node = lambda x: flatten([self.oobleaves(row, forest) for row in x])
        return [node(spill) for spill in spills]

    def predict(self):
        """Get aggregated prediction the leaf nodes generated on each spill"""
        yhat = lambda x: self.aggregator([node.prediction for node in x])
        return [yhat(group) for group in self.leaves]


class Forest(Diagnostics, Predict):
    """Random Forest class. Constructs either Classification Trees or
    Regression Trees based on the loss function. Classification Forests return
    the mode of the Classification Trees. Regression Forests return the mean
    of the Regression Trees.
    oob_err - Out of Bag Error
    """
    def __init__(self, modeldf, mingain=0, minrows=1, maxheight=10,
                 ntrees=10, nfeatures=1, conf=0.95):
        Tree.counter = 0  # reset tree counter prior to creating trees
        tparams = TreeParams(mingain, minrows, maxheight)
        self.configs = Settings(modeldf, tparams, ntrees, nfeatures)
        self.trees = {}
        self.boots = {}
        self.spills = {}
        self.build_forest(self.configs, conf)
        self.oob_err = OutofBag(self, conf).error
        trees = self.trees.values()
        nodes = flatten([list(tree.nodes.values()) for tree in trees])
        Diagnostics.__init__(self, nodes, modeldf.header)
        Predict.__init__(self, self, conf)

    def sample(self, data):
        """Create a bootstrapped replica of a data set. Returns the same
        number of rows as the original data set. Sample with replacement. Rows
        excluded from the sample go into spill used for out of bag error
        calculation"""
        pop_size = len(data)
        rownums = list(range(pop_size))
        n_size = int(0.99*pop_size)  # Ensure sample and spill have data
        bootpop = random.sample(rownums, k=n_size)  # Sample < Population
        samplerows = random.choices(bootpop, k=pop_size)
        spillrows = [x for x in rownums if x not in samplerows]
        return samplerows, spillrows

    def build_forest(self, settings, conf):
        """Build forest on bootstrap samples from training data"""
        data = settings.modeldf.train
        y = settings.modeldf.ycol
        kwargs = vars(settings.tparams)
        for idx in range(1, settings.ntrees + 1):
            samplerows, spillrows = self.sample(data)
            self.boots[idx] = [data[i] for i in samplerows]
            self.spills[idx] = [data[i] for i in spillrows]
            df = ListDF(self.boots[idx], settings.modeldf.header)
            bootdf = ModelDF(df, ycol=y, tst_prop=0)  # No split_train_test
            self.trees[idx] = Tree(bootdf, **kwargs, conf=conf)

    def __call__(self, index=1):
        pp.pprint(self.trees[index].__dict__.keys())
        return self.trees[index]

    def __repr__(self):
        ftype = {'gini': 'Classification Forest', 'var': 'Regression Forest'}
        return ftype[self.configs.loss.name]


# #########################prep funcs############
class ListDF:
    """A single dataframe object based on the python list object"""
    def __init__(self, data, header, sheetname=''):
        self.data = data
        self.header = header
        self.doctitle = sheetname
        self.numrows = len(data)
        self.numcols = len(header)
        self.set_colobjs(data, header)
        self.levels = self.set_levels()

    def set_colobjs(self, data, header):
        """Create column objects"""
        for i, title in enumerate(header):
            column = [row[i] for row in data]
            setattr(self, title, column)

    def unique_vals(self, col=None):
        """Find the unique values for a column in a dataset."""
        colname = self.header[col]
        column = getattr(self, colname)
        vals = set([x for x in column if x is not None])  # Exclude None
        return list(vals)

    def set_levels(self):
        """Calculate the number of unique items in each column"""
        uniques = [self.unique_vals(col=i) for i in range(self.numcols)]
        return list(map(len, uniques))

    def reset(self, data, header):
        """Update attributes of the class instance"""
        [delattr(self, title) for title in self.header]  # reset column objects
        self.data = data
        self.header = header
        self.numrows = len(data)
        self.numcols = len(header)
        self.set_colobjs(data, header)
        self.levels = self.set_levels()

    def names_or_nums(self, cols, colnames):
        """Process colnames into column numbers or return column numbers"""
        if colnames is not None:
            header = self.header
            cols = [header.index(x) for x in colnames if x in header]
        return cols

    def del_cols(self, cols=None, colnames=None):
        """Allow input of column numbers or column titles"""
        cols = self.names_or_nums(cols, colnames)
        keep = [i for i in range(self.numcols) if i not in cols]
        data = [[row[i] for i in keep] for row in self.data]
        header = [self.header[i] for i in keep]
        self.reset(data, header)

    def transform_col(self, fct, colname):
        col = self.header.index(colname)
        for row in self.data:
            row[col] = fct(row[col])
        self.reset(self.data, self.header)

    def sort_bycols(self, cols=None, colnames=None, descending=False):
        """Sort all rows based on column numbers or a column titles"""
        cols = self.names_or_nums(cols, colnames)
        sortby = lambda x: [x[col] for col in cols]
        self.data = sorted(self.data, key=sortby, reverse=descending)

    def export(self, filename='ListDataframe.csv', folder=os.getcwd()):
        data = self.data.copy()
        data.insert(0, self.header)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(data)
        txt = "{} rows exported to {}\n as {}\n"
        print(txt.format(self.numrows, folder, filename))

    def __call__(self, col_idx=0):
        return [row[col_idx] for row in self.data]

    def __repr__(self):
        display = copy.deepcopy(self.data)
        if self.numrows > 25:
            display = random.sample(self.data, k=25)
        display.insert(0, self.header)
        return neat_string(display)


class ModelDF(ListDF):
    """A dataframe object that seperates data into features and response"""
    def __init__(self, df, ycol=0, tst_prop=0.5):
        ListDF.__init__(self, df.data, df.header, df.doctitle)
        self.ycol = ycol
        self.xcols = [i for i, name in enumerate(self.header) if i != ycol]
        self.xdf = self.get_xdf(self.xcols)
        self.train, self.test = self.split_train_test(tst_prop)

    def get_xdf(self, xcols):
        xnames = [name for i, name in enumerate(self.header) if i in xcols]
        xdata = [[row[i] for i in xcols] for row in self.data]
        return ListDF(xdata, xnames)

    def split_train_test(self, tst_prop):
        """Split a data set on its rows into test and train data sets."""
        if tst_prop > 0:
            rownums = list(range(self.numrows))
            rowshuffle = random.sample(rownums, self.numrows)
            train_size = int(self.numrows - tst_prop*self.numrows)
            train = rowshuffle[:train_size]
            train_data = [r for i, r in enumerate(self.data) if i in train]
            test_data = [r for i, r in enumerate(self.data) if i not in train]
            return train_data, test_data
        return copy.deepcopy(self.data), copy.deepcopy(self.data)
