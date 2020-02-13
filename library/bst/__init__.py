"""Big __init__ file containing all functions"""
import os
import csv
import math
import numbers
import random
import statistics
from statistics import mean
import textwrap
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats

##Settings###############################
class Loss:
    """Controlling for misspelling of the names of the loss functions used
    to control the types of Decision Trees generated"""
    gini = "gini"
    var = "var"

class Settings:
    """Settings for the Binary Search Tree that persist between all of the
    modules in the library"""

    def __init__(self, data, ycol, header, loss_func, min_gain, min_rows,
                 forest, n_features, n_trees=1, sigdigits=3):
        self.y = ycol
        self.data = data
        self.loss_func = loss_func
        self.min_gain = min_gain
        self.min_rows = min_rows
        self.tree_type = classify_tree(loss_func)
        self.tree_idx = Tree.counter
        self.N = len(data)
        self.xcols = get_xcols(data, ycol)
        self.header = header
        self.yname = header[ycol]
        self.xnames = get_xnames(header, data, ycol)
        self.n_features = get_nfeatures(n_features, self.xnames, forest)
        self.forest = forest
        self.n_trees = n_trees
        self.sigdigits = sigdigits

def get_nfeatures(n_features, xnames, forest=False):
    featurevals = {True: n_features, False: len(xnames)}
    return featurevals[forest]

def get_xcols(data, ycol):
    colrange = range(len(data[0]))
    return [i for i in colrange if i != ycol]

def get_xnames(header, data, ycol):
    colrange = range(len(data[0]))
    return [header[i] for i in colrange if i != ycol]

def classify_tree(loss_func):
    tree_types = {Loss.var:'Regression', Loss.gini:'Classification'}
    return tree_types[loss_func]

##Tree Parts########################
class Question:
    """A question is used to partition a dataset into true and false answers
    """
    def __init__(self, col_num, value, header):
        self.header = header
        self.col_num = col_num
        self.value = value
        self.col_name = header[col_num]

    def match(self, row):
        """Compare the feature value in a row to the
        feature value in question."""
        val = row[self.col_num]
        if is_numeric(val):
            return val >= self.value
        if type(val) == str:
            return val == self.value

    def __repr__(self):
        condition = "=="
        val = self.value
        if is_numeric(self.value):
            val = round(self.value, 3)
            condition = ">="

        str_val = str(val)
        return "Is {} {} {}?".format(self.col_name, condition, str_val)

class Leaf:
    """A Leaf node predicts the response based on the data. It predicts
    mode in the case of Classification Tree and mean in the case of
    Regression Tree"""

    def __init__(self, ycol, idx, score, gain, data, settings):
        self.ycol = ycol
        self.ytrue = pull_y(data, ycol)
        self.index = idx
        self.score = score
        self.gain = gain
        self.data = data
        self.n = len(data)
        self.tree = settings.tree_idx
        self.prediction = predict(settings.loss_func, self.ytrue)

def pull_y(data, ycol):
    return [row[ycol] for row in data]

def predict(loss_func, y_true):
    mode = lambda x: max(set(x), key=x.count)
    func = {Loss.gini: mode, Loss.var: scipy.mean}
    return func[loss_func](y_true)

class Decision:
    """A Decision Node asks a question. The question results in a true branch
    and a false branch in response to the question."""

    def __init__(self, question, data, left, right, idx, score, gain, settings):
        self.ycol = settings.y
        self.question = question
        self.data = data
        self.true_branch = left
        self.false_branch = right
        self.index = idx
        self.n = len(data)
        self.score = score
        self.gain = gain
        self.tree = settings.tree_idx

##################################Build Tree#################################
import gc

class BinarySearch:
    """Method class for tree construction. Performs binary search.
    Binary refers to how the features are seperated using Questions.
    """
    def gen_breakpts(data, colnum):
        """Find all the candidate values for splitting the data on a
        given feature into groups. Return k-1 candidates for string variables.
        """
        unique = unique_vals(data, colnum)
        if any([type(elm) == str for elm in unique]):
            results = unique[1:]
        elif all([is_numeric(elm) for elm in unique]):
            cuts = range(0, len(unique), 2)
            pair = lambda x: [x[i:i + 2] for i in cuts if len(x[i:i + 2]) == 2]
            pairs = pair(unique)
            results = [mean(group) for group in pairs]
        return results

    def decide(parent_idx, child_idx):
        """Find the direction of the split that took place based on the index
        value of the child. Indexes are integers"""
        if child_idx == 2 * parent_idx:
            decision = 'Yes'
        elif child_idx == 2 * parent_idx + 1:
            decision = 'No'
        return decision

    def gen_branch(idx, ancestors=None):
        """Recursive function to find the ancestors of a particular node
        ancestors = None is necessary to keep the recursive function from
        storing ancestors between function calls which resuts in duplicates
        """
        if ancestors == None:
            ancestors = []
        parent = math.floor(idx/2)
        if parent == 0:
            ancestors.append(idx)
            return sorted(ancestors)
        ancestors.append(idx) #Due to recursion its idx and not parent
        return BinarySearch.gen_branch(parent, ancestors)

    def get_answers(branch):
        """Record all of the answers to questions that occur on a branch.
        Yes -> 2k, No -> 2k+1 where k is the preceding node
        """
        func = BinarySearch.decide
        parent_child = [branch[i_: i_ + 2] for i_ in range(len(branch))][:-1]
        return [func(pair[0], pair[1]) for pair in parent_child]

    def attach_nodes(tree_idx):
        """add Leaf and Decision objects to the tree"""
        treenodes = {}
        for obj in gc.get_objects():
            if isinstance(obj, (Leaf, Decision)) and obj.tree == tree_idx:
                treenodes[obj.index] = obj
        return sortdict(treenodes, sortbykeys=True, descending=False)

    def assign_objects(objtype, nodes):
        """Function to create the leaf and decision list of indexes"""
        return [idx for idx, x in nodes.items() if isinstance(x, objtype)]

    def find_missed(nodes):
        """Find the nodes that are not stored in the object class dictionaries
        yet should exist on the tree. The branch leading to the existing node
        are all of the nodes that should exist on the tree"""
        maxnode = max(nodes.keys())
        search = [idx for idx in range(1, maxnode) if idx not in nodes.keys()]
        missing = {}
        for key in nodes.keys():
            branch = BinarySearch.gen_branch(key)
            for idx in branch:
                if idx in search:
                    parent = math.floor(idx/2)
                    conclusion = BinarySearch.decide(parent, idx)
                    if conclusion == 'Yes':
                        missing[idx] = nodes[parent].true_branch
                    elif conclusion == 'No':
                        missing[idx] = nodes[parent].false_branch
        return missing

    def bestsplit(settings, data, ycol, idx):
        """Find the best question to ask by iterating over every feature
        and calculating the information gain.
        """
        score = Score(settings).node_score(data, ycol)
        gain = settings.min_gain # keep track of the best info gain
        question = None # keep track of the value that produced it

        #feature subset applies to Random Forest
        features = random.sample(settings.xcols, settings.n_features)
        features.sort()

        for fcol in features:
            breakpts = BinarySearch.gen_breakpts(data, fcol)
            if len(breakpts) == 0:
                continue
            for val in breakpts:
                split = Split(fcol, val, ycol, idx, data, settings)
                if not split.enough:
                    continue
                if split.info_gain > gain:
                    gain, question = split.info_gain, split.question

        return gain, question, score

    def build_tree(settings, data, ycol, idx=1):
        """Builds the tree.
        Base case: no further info gain. Since we can ask no further questions,
        we'll return a leaf.
        Otherwise: Continue recursively down both branches. Return a Decision
        node. The Decision node records the question and both branches.
        """
        gain, question, score = BinarySearch.bestsplit(settings, data, ycol, idx)
        if question == None:
            leaf = Leaf(ycol, idx, score, gain, data, settings)
            return leaf

        args = (question.col_num, question.value)
        split = Split(*args, ycol, idx, data, settings)

        lft_idx, rgt_idx = split.child_true, split.child_false
        true_rows, false_rows = split.true_rows, split.false_rows
        left = BinarySearch.build_tree(settings, true_rows, ycol, lft_idx)
        right = BinarySearch.build_tree(settings, false_rows, ycol, rgt_idx)

        args = (question, data, left, right, idx, score, gain, settings)
        return Decision(*args)

    def calc_height(nodes):
        """Size of a binary tree of height, h is 2^h - 1
        Height = log2(n+1) where n is the max node"""
        idx = max(nodes, default=1)
        h = round(math.log2(idx + 1))
        return h

class Tree:
    """Decision Tree class.  Classification Trees and Regression Trees
    depend on the loss function type."""
    counter = 0

    def __init__(self, data, ycol, header, lossfunc,
                 min_gain=0, min_rows=1, forest=False, n_features=1):
        Tree.counter += 1

        args = (data, ycol, header, lossfunc, min_gain, min_rows)
        self.configs = Settings(*args, forest, n_features)

        self.model = BinarySearch.build_tree(self.configs, data, ycol)
        self.nodes = BinarySearch.attach_nodes(self.configs.tree_idx)
        self.height = BinarySearch.calc_height(self.nodes)
        self.leaves = BinarySearch.assign_objects(Leaf, self.nodes)
        self.decisions = BinarySearch.assign_objects(Decision, self.nodes)

    def __call__(self, index=1):
        return self.nodes[index]

    def __str__(self):
        attr = "{}".format([x for x in dir(self) if "__" not in x])
        return "Tree Class attributes\n {}".format(wrap(attr))

class Split:
    """Split rows at a specific value of a feature"""

    def __init__(self, col, value, ycol, idx, data, settings):
        self.question = Question(col, value, settings.header)
        self.true_rows, self.false_rows = self.partition(data, self.question)
        self.child_true, self.child_false = self.child_idxs(idx)
        self.size = self.calc_size(self.true_rows, self.false_rows)

        args = (self.true_rows, self.false_rows)
        self.info_gain = self.calc_gain(*args, data, ycol, settings)
        self.enough = self.valid(settings)

    def child_idxs(self, idx):
        """True answers to the question result in 2*x False answers to the
        question result in 2*x+1"""
        return 2*idx, 2*idx + 1 #Left child, Right child

    def partition(self, rows, question):
        """Partition dataset by answer to the class: question into
        true rows and false rows"""
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    def calc_size(self, true_rows, false_rows):
        nleft = len(self.true_rows)
        nright = len(self.false_rows)
        return min(nleft, nright)

    def calc_gain(self, left, right, data, ycol, settings):
        """Determine the information gained from the split"""
        redux = Score(settings).wgtd_score(left, right, ycol)
        score = Score(settings).node_score(data, ycol)
        weight = len(data)/settings.N
        return weight * (score - redux)

    def valid(self, settings):
        """Determine if the split is valid based on the input settings"""
        big_enough = self.size >= settings.min_rows
        enough_gain = self.info_gain >= settings.min_gain
        validsplit = all([big_enough, enough_gain])
        return validsplit

class Score:
    """Calculation methods for assessing impurity of the response column"""
    def __init__(self, settings):
        self.node_score = self.getfunc(settings)

    def getfunc(self, settings):
        """The function used to calculate impurity of a node"""
        func = {Loss.gini: self.gini, Loss.var: self.var}
        return func[settings.loss_func]

    def gini(self, data, resp_col):
        """Calculate the Gini Impurity for a list of rows"""
        counts = class_counts(data, resp_col)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / len(data)
            impurity -= prob_of_lbl**2
        return impurity

    def var(self, data, resp_col):
        y_stat = [row[resp_col] for row in data]
        if len(y_stat) > 1:
            return statistics.variance(y_stat)
        else:
            return 0

    def wgtd_score(self, left, right, ycol):
        """Weighted impurity based on size of the child nodes"""
        prop = len(left) / (len(left) + len(right))
        impurity_left = self.node_score(left, ycol)
        impurity_right = self.node_score(right, ycol)
        return prop * impurity_left + (1 - prop) * impurity_right

########################Diagnostics########################
class Diagnostics:
    """Method class for calculating reports on the Tree or Forest objects"""
    def make_leafmatrix(obj, settings):
        """Identifies the features used for making predictions on a leaf.
        Function returns an indicator vector for all of the features"""
        header = flatten(["Tree", "Leaf", "Rows", "Score", settings.xnames])
        matrix = []

        for treeidx in range(settings.n_trees):
            nodes = Diagnostics.getnodes(obj, treeidx)
            for idx, node in nodes.items():
                if isinstance(node, Leaf):
                    feature_row = [0 for i in settings.header]
                    path = BinarySearch.gen_branch(idx)[:-1]
                    for idx in path:
                        if idx in nodes.keys():
                            feature_row[nodes[idx].question.col_num] = 1
                    feature_row.pop(settings.y)
                    row = [treeidx, node.index, node.n, node.score, feature_row]
                    matrix.append(flatten(row))

        return ListDF(matrix, header)

    def mark_leafclusters(obj, settings):
        """Add Leaf column to the training data"""
        header = flatten(["Tree", "Leaf_idx", settings.header])
        marked_data = []
        for treeidx in range(settings.n_trees):
            nodes = Diagnostics.getnodes(obj, treeidx)
            for idx, node in nodes.items():
                if isinstance(node, Leaf):
                    for row in node.data:
                        labeled_row = flatten([treeidx, idx, row])
                        marked_data.append(labeled_row)

        return ListDF(marked_data, header)

    def tally_gain(nodes, importance):
        """Total the information gained on each feature in Tree.nodes"""
        for idx, node in nodes.items():
            if isinstance(node, Decision):
                feature = node.question.col_name
                importance[feature] += node.gain
        return importance

    def score_features(obj, settings, standardize=True):
        """Find the feature importance in the tree. In a Tree importance is
        the total information gain. In a Forest importance is the average
        information gain across all Trees.
        """
        importance = {x:0 for x in settings.header}
        for treeidx in range(settings.n_trees):
            nodes = Diagnostics.getnodes(obj, treeidx)
            importance = Diagnostics.tally_gain(nodes, importance)
        importance = {k:v/settings.n_trees for k, v in importance.items()}

        scores = {}
        for key, val in importance.items():
            if val > 0:
                scores[key] = val

        if standardize:
            vals = sorted(scores.values(), reverse=True)
            key = lambda x: get_key(x, scores)
            percents = {key(v):v/sum(vals) for v in vals}
            return percents

        return scores

    def make_nodesummary(obj, settings):
        """Summary of the nodes in the tree"""
        header = ["Tree", "Index", "Type", "Question", "Rows", "Score", "Gain"]
        data = []
        for treeidx in range(settings.n_trees):
            nodes = Diagnostics.getnodes(obj, treeidx)
            for idx, node in nodes.items():
                if isinstance(node, Leaf):
                    half_1st = [treeidx, node.index, "Leaf", ""]
                    half_2nd = [node.n, node.score, node.gain]
                    data.append(flatten(half_1st + half_2nd))
                if isinstance(node, Decision):
                    third_1st = [treeidx, node.index, "Decision"]
                    third_2nd = [str(node.question), node.n, node.score]
                    third_3rd = [node.gain]
                    data.append(flatten(third_1st + third_2nd + third_3rd))

        summaryDF = ListDF(data, header)
        summaryDF.sort_col(colname="Index")
        return summaryDF

    def classify(row, node):
        """Recursive function for returning Leaf node.
        Base case: reach a Leaf node
        Recursion: traverse Decision Nodes along branches until Base Case
        """
        if isinstance(node, Leaf):
            return node
        if node.question.match(row):
            return Diagnostics.classify(row, node.true_branch)
        else:
            return Diagnostics.classify(row, node.false_branch)

    def getmodel(obj, treeidx=0):
        """The model allows for recursive searching i.e. the application of
        the classify function"""
        if isinstance(obj, Forest):
            return obj.trees[treeidx]['tree'].model
        if isinstance(obj, Tree):
            return obj.model

    def getfunc(obj):
        """The function used to aggregate predictions in a forest"""
        mode = lambda x: max(set(x), key=x.count)
        func = {Loss.gini: mode, Loss.var: scipy.mean}
        return func[obj.configs.loss_func]

    def getnodes(obj, treeidx=0):
        if obj.configs.forest:
            return obj.trees[treeidx]['tree'].nodes
        if isinstance(obj, Tree):
            return obj.nodes

    def genleaves(obj, testdata, settings):
        """Generate the model estimates on each row of test data"""
        leaf_nodes = []
        for row in testdata:
            noderow = []
            for treeidx in range(settings.n_trees):
                model = Diagnostics.getmodel(obj, treeidx)
                noderow.append(Diagnostics.classify(row, model))
            leaf_nodes.append(noderow)
        return leaf_nodes

    def makepredictions(leafnodes, func):
        """Use the function to compile the prediction on each leafnode.
        For a Forest there are multiple nodes in each row of leafnodes.
        For a Tree there is one node in each row of leafnodes."""
        predictions = []
        for row in leafnodes:
            prediction = func([node.prediction for node in row])
            predictions.append(prediction)
        return predictions

    def getleaf_idx(leafnodes):
        """Finds the index used to make a prediction on a row of data"""
        leaf_pos = []
        for row in leafnodes:
            idx = [node.index for node in row]
            if len(idx) == 0:
                idx = idx[0]
            leaf_pos.append(idx)
        return leaf_pos

    def Xmatrix(obj, data):
        feat_matrix = [[row[x] for x in obj.configs.xcols] for row in data]
        return ListDF(feat_matrix, obj.configs.xnames)

    def make_pred_df(obj, testdata, leaves, yhat, ytrue, intervals):
        pheader = ["Leaf", "y_hat", obj.configs.yname, "Interval"]
        xDF = Diagnostics.Xmatrix(obj, testdata)
        pdata = [leaves, yhat, ytrue, intervals]
        pDF = ListDF(pdata, pheader)
        pDF.paste(xDF.data, xDF.header)
        return pDF

class GOF:
    """Method class to contain goodness of fit calculation measures"""
    def mse(ytrue, yhat, loss_func):
        mse = None
        if loss_func == Loss.var:
            mse = mean_squared_error(ytrue, yhat)
        return mse

    def rsq(ytrue, yhat, loss_func):
        rsq = None
        if loss_func == Loss.var:
            rsq = r2_score(ytrue, yhat)
        return rsq

    def adjrsq(ytrue, yhat, n_, p_, loss_func):
        adjrsq = None
        if loss_func == Loss.var:
            rsq = r2_score(ytrue, yhat)
            adjrsq = 1 - (1 - rsq)*(n_ - 1)/(n_ - p_ - 1)
        return adjrsq

    def calc_conf_int(y_list, conf):
        """For the variance loss function, return the mean of the y feature
        and a confidence interval around the mean."""
        mu, se = scipy.mean(y_list), 0
        if len(y_list) > 1:
            se = scipy.stats.sem(y_list)
        if se == 0:
            return 0, 0

        low, high = scipy.stats.norm.interval(conf, mu, se)
        return low, high

    def conf_int(leafnodes, loss_func, conf):
        """Confidence interval varies based on the actual response values
        of the leaf node(Tree) or leaf nodes(Forest)"""
        if loss_func == Loss.var:
            intervals = []
            for row in leafnodes:
                yrow = flatten([node.ytrue for node in row])
                interval = GOF.calc_conf_int(yrow, conf)
                intervals.append(interval)
            return intervals
        if loss_func == Loss.gini:
            return [[None, None] for row in leafnodes]

    def err(intervals, ytrue, yhat, loss_func):
        """RegressionTree: Determine when intervals contains ytrue
        ClassificationTree: Determine when yhat = ytrue
        """
        if loss_func == Loss.var:
#            btwn = lambda x, pair: pair[0] <= x <= pair[1]
#            accuracy = [btwn(ytrue[i], pair) for i, pair in enumerate(intervals)]
#            return accuracy.count(False)/len(ytrue)
            return mean_squared_error(ytrue, yhat)
        if loss_func == Loss.gini:
            accuracy = [ytrue[i] == yhat[i] for i in range(len(ytrue))]
            return accuracy.count(False)/len(ytrue)

class Summary:
    """Diagnostic reports of the binary search object"""
    def __init__(self, obj):
        settings = obj.configs
        self.obj = obj
        self.leafmatrix = Diagnostics.make_leafmatrix(obj, settings)
        self.leafclusters = Diagnostics.mark_leafclusters(obj, settings)
        self.featureimportance = Diagnostics.score_features(obj, settings)
        self.nodesummary = Diagnostics.make_nodesummary(obj, settings)

    def traverse_to(self, treeidx=0, nodeidx=1):
        """Given a node in the tree recreate the decisions that
        occur to reach the node. """
        tree = self.obj
        if self.obj.configs.forest:
            tree = self.obj.trees[treeidx]['tree']

        branch = BinarySearch.gen_branch(nodeidx)
        answers = BinarySearch.get_answers(branch)
        path = [pair for pair in zip(branch, answers)]

        print("Traversing the tree to node {}...".format(nodeidx))
        for pair in path:
            idx, answer = pair[0], pair[1]
            score = round(tree(idx).score, tree.configs.sigdigits)
            print("Node({}) Rows {} Score {}".format(idx, tree(idx).n, score))
            if isinstance(tree(idx), Decision):
                print("\t{} {}".format(tree(idx).question, answer))
            elif isinstance(tree(idx), Leaf):
                prediction = tree(idx).prediction
                print("\tLeaf Node Prediction {}\n".format(prediction))

    def print_tree(self, Tree, spacing=""):
        """print_tree(my_tree.model)"""
        if isinstance(Tree.model, Leaf):
            print(spacing + "Predict", Tree.model.prediction)
            return

        print(spacing + str(Tree.model.question))
        print(spacing + '--> True:')
        self.print_tree(Tree.model.true_branch, spacing + "  ")
        print(spacing + '--> False:')
        self.print_tree(Tree.model.false_branch, spacing + "  ")

#########################Predict##################################
class Predict:
    """Generates predictions from Decision Tree or Random Forest object"""
    def __init__(self, obj, testdata, conf=0.95):
        settings = obj.configs
        self.n = len(testdata)
        self.loss_func = obj.configs.loss_func
        self.numpredictors = len(Diagnostics.score_features(obj, settings))
        self.leafnodes = Diagnostics.genleaves(obj, testdata, settings)
        self.leaves = Diagnostics.getleaf_idx(self.leafnodes)
        self.predictmethod = Diagnostics.getfunc(obj)
        args = (self.leafnodes, self.predictmethod)
        self.yhat = Diagnostics.makepredictions(*args)
        self.ytrue = [row[obj.configs.y] for row in testdata]
        self.conf_intervals = GOF.conf_int(self.leafnodes, self.loss_func, conf)
        args = (self.conf_intervals, self.ytrue, self.yhat, self.loss_func)
        self.error = GOF.err(*args)
        self.rsq = GOF.rsq(self.ytrue, self.yhat, self.loss_func)
        args = (self.ytrue, self.yhat, self.n, self.numpredictors)
        self.adj_rsq = GOF.adjrsq(*args, self.loss_func)
        self.mse = GOF.mse(self.ytrue, self.yhat, self.loss_func)
        self.msg = self.display_gof(settings)
        args = (obj, testdata, self.leaves, self.yhat, self.ytrue)
        self.df = Diagnostics.make_pred_df(*args, self.conf_intervals)

    def display_gof(self, settings):
        """Displays the error and goodness of fit attributes """
        msg_basic = "Testing on {} rows\n".format(self.n)

        if settings.loss_func == Loss.var:
            txt = "RSq {0[0]:.0%} AdjRSq {0[1]:.0%} MSE {0[2]:.2f}\n"
            msg_reg = txt.format([self.rsq, self.adj_rsq, self.mse])
            msg = msg_basic + "\n" + msg_reg

        elif self.loss_func == Loss.gini:
             msg_class = "Accuracy {:.2%}\n".format(1-self.error)
             msg = msg_basic + "\n" + msg_class

        return msg

#####################Forest############################################
import copy

class Bootstrap:
    """Method class for construction of Random Forest"""
    def colmin(xlist, colnum):
        """Find the row containing the minimum value of a column"""
        col = [row[colnum] for row in xlist]
        the_min = min(col)
        return [row for row in xlist if row[colnum] == the_min]

    def joins_a(a_data, b_data):
        """Returns a dictionary object. Where dict.value = 1 results in an
        inner join of A and B. Where dict.value = 0 results in a Left Outer
        Exclude B.
        """
        in_a = {str(row):0 for row in a_data}
        for row in b_data:
            if str(row) in in_a:
                in_a[str(row)] += 1
        return in_a

    def left_outer_exclude_b(a_data, b_data):
        """Similar to a sql left outer join excluding B this function finds all
        rows in a that are not in b."""
        in_a = Bootstrap.joins_a(a_data, b_data)
        return [row for row in a_data if in_a[str(row)] == 0]

    def sample(data):
        """Create a bootstrapped replica of a data set. Function returns the
        same number of rows as the original data set. Bootstrapping samples
        rows from the original data set with replacement."""
        output = []
        for i in range(len(data)):
            rand_data = [(random.random(), row) for i, row in enumerate(data)]
            the_min_row = Bootstrap.colmin(rand_data, 0)
            the_min_row = the_min_row[0][1] #[0] unlists [1] select data row
            output.append(the_min_row)
        return output

    def build_forest(settings, n_trees, data, ycol):
        trees = {}
        for idx in range(n_trees):
            boot = Bootstrap.sample(data)
            spill = Bootstrap.left_outer_exclude_b(data, boot)
            args1 = [boot, ycol, settings.header, settings.loss_func]
            args2 = [settings.min_gain, settings.min_rows, settings.forest]
            args = tuple(args1 + args2)
            tree = Tree(*args, settings.n_features)
            trees[idx] = {'data': boot, 'out_of_bag': spill, 'tree': tree}
        return trees

    def oob_forest(forest, treeidx):
        """Return the subforest that is out of the bag of the tree referenced"""
        subforest = copy.deepcopy(forest)
        trees = {k:v for k, v in forest.trees.items() if k != treeidx}
        vals = trees.values()
        subforest.trees = {k:v for k, v in enumerate(vals)} #reset dict
        subforest.configs.n_trees = len(subforest.trees)
        return subforest

    def calc_oob_error(forest, n_trees):
        """Calculate out of bag error for each sample"""
        err_rate = 0
        for idx in range(n_trees):
            subforest = Bootstrap.oob_forest(forest, idx)
            testdata = forest.trees[idx]['out_of_bag']
            prediction = Predict(subforest, testdata)
            err_rate += prediction.error
        return err_rate/n_trees

class Forest:
    """Random Forest class. Constructs either Classification Trees or
    Regression Trees based on the loss function. Classification Forests return
    the mode of the Classification Trees. Regression Forests return the mean
    of the Regression Trees.
    oob_err - Out of Bag Error
    """

    def __init__(self, data, ycol, header, lossfunc, min_gain=0, min_rows=1,
                 forest=True, n_features=1, n_trees=10):
        args = (data, ycol, header, lossfunc, min_gain, min_rows)
        self.configs = Settings(*args, forest, n_features, n_trees)
        self.trees = Bootstrap.build_forest(self.configs, n_trees, data, ycol)
        self.oob_err = Bootstrap.calc_oob_error(self, n_trees)

    def __call__(self, index=0):
        return self.trees[index]['tree']

    def __str__(self):
        attr = "{}".format([x for x in dir(self) if "__" not in x])
        return "Forest Class attributes\n {}".format(wrap(attr))


##########################prep funcs############
class ConfigStub:
    '''
    There once was a library called config, it had these raw settings. This class
    helps so we don't have to fix all that config code right now.
    '''
    def __init__(self):
        self.MAX_WIDTH = 100
        self.ROWS_TO_DISPLAY = 10
        self.SIG_DIGITS = 2

config = ConfigStub() # pylint: disable=invalid-name

def scrub(sheet):
    """ Function cleans an individual csv sheet.
    1. Convert strings to numeric where appropriate
    2. Remove blank rows
    3. Find the header row
    """
    new_sheet = str_to_num(sheet)
    new_sheet = remove_blank_rows(new_sheet)
    new_sheet = new_sheet[find_header(new_sheet):]
    return new_sheet

def find_header(sheet):
    """
    The idea is that the header row is made up of only strings and it is a
    complete row, i.e. number of string columns match the maximum number of
    columns in the dataset. Rows could be blank or incomplete so the function
    measures the length of each row.
    """
    cols = num_cols(sheet)
    for row_i in range(len(sheet)):
        row = sheet[row_i]
        row_length = range(len(row))
        col_cnt = sum(1 for i in row_length if type(row[i]) == str)
        if col_cnt == cols:
            header_row = row_i
            break
        else:
            header_row = 0
    return header_row

def remove_blank_rows(sheet):
    """ Blank rows are determined by the length of the row. Even a single
    digit will still register as a positive length.
    """
    del_rows = [i for i in range(len(sheet)) if len(sheet[i]) == 0]
    new_sheet = [sheet[i] for i in range(len(sheet)) if i not in del_rows]
    return new_sheet

def str_to_num(sheet):
    """
    Turn strings into numbers including percents wherever possible.
    Element cannot be empty string. -1 means last element of list
    """
    for i in range(len(sheet)):
        for j in range(len(sheet[i])):
            elm = sheet[i][j]
            if type(elm) == str and len(elm) > 0:
                try:
                    sheet[i][j] = float(elm.replace(",", ""))
                except ValueError:
                    pass
                if elm[-1] == "%":
                    sheet[i][j] = float(elm[:-1])/100.
    return sheet

def delete_cols(sheet, select_cols):
    """Function deletes the columns of a list. Sorting the select_cols in
    reverse order is essential to the success of the function. Once a column
    is deleted the number of columns available changes.
    """
    try:
        select_cols.sort(reverse=True)
        for row in sheet:
            for i in select_cols:
                del row[i]
    except:
        print("Columns selected for deletion must be a list")
    return sheet

def remove_blank_cols(sheet):
    """Function determines how sparse the column is across all rows. Columns
    that have greater sparseness than the percentage threshold are deleted.
    """
    cols = num_cols(sheet)
    rows = len(sheet)
    columnloss = [0 for i in range(cols)]

    for i in range(cols):
        for row in sheet:
            if len(str(row[i])) == 0:
                columnloss[i] += 1/rows
    emptycols = [i for i in range(cols) if columnloss[i] > 0.9]
    cleanedsheet = delete_cols(sheet, emptycols)
    return cleanedsheet

def unstack(sheet, col_to_unstack, grp_col):
    """Function unstacks a column using the grp_col. Each unique
    element of the grp_col becomes a new column. The values in the
    column to unstack are spread to each new group column.
    """
    groups = [row[grp_col] for row in sheet[1:]]
    colnames = list(set(groups))

    for colname in colnames:
        sheet[0].append(colname)
        for row in sheet[1:]:
            if row[grp_col] == colname:
                row.append(row[col_to_unstack])
            else:
                row.append('')

    delete_cols(sheet, [col_to_unstack, grp_col])
    return sheet

class Folder:
    """Ingests all csv files in a specified folder. The object then stores
    multiple attributes of every file. """
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.files = [filename for filename in os.listdir(dirpath)]
        self.sheets = [loadcsv(dirpath, filename) \
                       for filename in os.listdir(dirpath)]
        self.filecount = len(self.files)
        self.sheets = [scrub(sheet) for sheet in self.sheets]
        self.cols = [num_cols(sheet) for sheet in self.sheets]
        self.rows = [len(sheet) for sheet in self.sheets]
        self.headers = [sheet[0] for sheet in self.sheets]

    def __call__(self, fileindex=0):
        sheet = self.sheets[fileindex][1:]
        header = self.headers[fileindex]
        sheetname = self.files[fileindex]
        print("%d files in the folder %s" % (self.filecount, self.dirpath))
        print(print_sheet(sheet, header, sheetname))

    def collate(self):
        collated = collate_sheets(self.sheets)
        collated = remove_blank_cols(collated)
        return collated

    def export(self):
        exportcsv(filename=self.files[0], data=self.sheets,
                  folder=os.getcwd())

    def to_list_df(self, fileindex=0):
        """Send a file to the ListDF class"""
        return ListDF(self.sheets[fileindex][1:], self.headers[fileindex],
                      self.files[fileindex])

class ListDF:
    """a single data table object"""
    def __init__(self, rows, header, sheetname=''):
        self.data = rows
        self.header = header
        self.num_rows = len(rows)
        self.num_cols = num_cols(rows)
        self.doctitle = sheetname
        for i in range(len(header)):
            title = header[i]
            column = [row[i] for row in rows]
            setattr(self, title, column)

    def __call__(self, col_idx=0):
        return [row[col_idx] for row in self.data]

    def add_col(self, col_data, col_name, insert_spot=-1):
        """Insert a new column into the data table"""
        for i, row in enumerate(self.data):
            row.insert(insert_spot, col_data[i])
        self.header.insert(insert_spot, col_name)
        self.__init__(self.data, self.header)

    def paste(self, data, names):
        """Paste data in front of data frame"""
        rows = [pair[0] + pair[1] for pair in zip(data, self.data)]
        header = names + self.header
        self.__init__(rows, header)

    def subset(self, col):
        """Create a subset ListDF from a column list of numbers"""
        rows = [[row[idx] for idx in col] for row in self.data]
        header = [self.header[idx] for idx in col]
        return ListDF(rows, header)

    def transform_col(self, fct, col_name):
        i = self.header.index(col_name)
        for row in self.data:
            row[i] = fct(row[i])
        self.__init__(self.data, self.header)

    def del_col(self, col=None, colname=''):
        """Allow input of a column number or a column title"""
        if colname in self.header:
            col = self.header.index(colname)
        #todo: this is a big issue to look into
        [row.pop(col) for row in self.data]
        self.header.pop(col)
        self.__init__(self.data, self.header)

    def sort_col(self, col=None, colname='', descending=False):
        """Sort all rows based on a column number or a column title"""
        if colname in self.header:
            col = self.header.index(colname)

        data = sorted(self.data, key=lambda x: x[col], reverse=descending)
        self.__init__(data, self.header)

    def col_order(self, cols=[], colnames=''):
        """Sort data based on column number or a column title"""
        if set(colnames).issubset(set(self.header)):
            cols = [self.header.index(colname) for colname in colnames]

        data = [self.data[col] for col in cols]
        header = [self.header[col] for col in cols]
        self.__init__(data, header)

    def export(self, filename='ListDataframe.csv', folder=os.getcwd()):
        rows = self.data.copy()
        if rows[0] == self.header:
            rows = self.data[1:]
        else:
            rows.insert(0, self.header)
        exportcsv(filename, rows, folder)
        print("{} rows exported to {}\n as {}\n".format(self.num_rows, folder, filename))

    def __repr__(self):
        return print_sheet(self.data, self.header, self.doctitle)

def collate_sheets(sheets):
    """Combine sheets together provided that the header rows all match.
    Function input is the class object loadedfiles
    """
    header = sheets[0][0]
    row_count = 0
    print("Sheets merging...")
    for i in range(len(sheets)):
        sheet = sheets[i]
        if i == 0:
            rows = len(sheet)
            combined = sheet
            row_count += rows
        else:
            if sheet[0] == header:
                rows = len(sheet[1:])
                combined.extend(sheet[1:])
                row_count += rows
            else:
                print("Sheet %d does not match" % i)
        print("Sheet %03d Rows %03d %03d" % (i, rows, row_count))
    print("Sheet All Rows %03d\n" % row_count)
    return combined

def loadcsv(folder, filename):
    """folder is a filepath. filename includes the extension of the file.
    The function reads in csv files and returns a list.
    """
    filepath = os.path.join(folder, filename)
    try:
        with open(filepath, newline='') as csvfile:
            csv_list = list(csv.reader(csvfile))
            return csv_list
    except:
        print("Something went wrong with %s" % filename)

def exportcsv(filename, data, folder=os.getcwd()):
    """folder is a filepath. filename includes the extension of the file.
    The function exports a list as a csv.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerows(data)

class CsvHelper:

    def __init__(self):
        self.name = "hi"

    def loadcsv(self, folder, filename):
        """folder is a filepath. filename includes the extension of the file.
        The function reads in csv files and returns a list.
        """
        filepath = os.path.join(folder, filename)
        #try:
        with open(filepath, newline='') as csvfile:
            csv_list = list(csv.reader(csvfile))
            return csv_list
        #except:
        #    print("Something went wrong with %s" % filename)

    def exportcsv(self, filename, data, folder=os.getcwd()):
        """folder is a filepath. filename includes the extension of the file.
        The function exports a list as a csv.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(data)

    def csv_to_df(self, data):
        scrub(data)
        #col = num_cols(sheet)
        #row = len(sheet)
        #header = sheet[0]
        return ListDF(data[1:], data[0])

def find_root(name, path):
    for root, dirs, files in os.walk(path):
        if name in root:
            return root

def find_subroot(partial_path, path):
    """Returns the root containing all of the names in the partial path"""
    try:
        names = partial_path.split('/')
    except SyntaxError:
        print("Oops! Only use forwardslash('\\') for your input string.")
    possible_roots = []
    for name in names:
        possible_roots.append(find_root(name, path))
    for root in possible_roots:
        if all(name in root for name in names):
            return root

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def make_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

def shorten_line(line, max_width=config.MAX_WIDTH):
    """For more attractive printing on the screen. Shortens the line
    displayed. Function breaks a line into two parts. The first part
    accepts the first 2/3 of characters. The second part accepts the
    last 2/3 of characters counting from the end of the string."""
    newline = [round(x, config.SIG_DIGITS) if isinstance(x, float) else x
               for x in line]
    linetext = ", ".join(str(x) for x in newline)
    line_length = len(linetext)
    linedict = {k:v for k, v in enumerate(newline)}
    if line_length > max_width:
        width_tally = int(2/3*max_width+4)
        a_short = textwrap.shorten(linetext, width=width_tally, placeholder='....')

        keep = []
        for k in sorted(linedict, reverse=True):
            width_tally += len(str(linedict[k]))
            if width_tally <= max_width:
                keep.append(k)
        keep.sort()

        b_join = ", ".join(str(linedict[k]) for k in keep)
        return a_short + b_join
    else:
        return newline

def wrap(text, width=75):
    wrapper = textwrap.TextWrapper(width)
    word_list = wrapper.wrap(text)
    row = ' '.join([str(elem + '\n') for elem in word_list])
    return '%s' % row

def print_sheet(sheet, header, sheetname='', cut=config.ROWS_TO_DISPLAY):
    """Function prints lists in matrix format for specified number of rows
    """
    num_rows = len(sheet)
    num_cols_loc = num_cols(sheet)
    string_list = []

    if len(sheetname) > 0:
        string_list.append("%s" % sheetname)
    string_list.append("%d Rows X %d Columns\n" % (num_rows, num_cols_loc))
    string_list.append("Header: %s" % shorten_line(header))

    for row in range(num_rows):
        line = shorten_line(sheet[row])
        if row <= cut or row >= num_rows-cut:
            string_list.append("Row %02d: %s" % (row, line))
        elif row == cut + 1:
            string_list.append(" ".join("..." for x in sheet[row]))

    string_list.append("\n")
    return "\n".join(string_list)

def print_list(data, cut=config.ROWS_TO_DISPLAY):
    """Function prints lists in matrix format for specified number of rows
    """
    num_rows = len(data)
    string_list = []

    for row in range(num_rows):
        line = shorten_line(data[row])[0]
        if row <= cut or row >= num_rows-cut:
            string_list.append(line)
        elif row == cut + 1:
            string_list.append("....")

    return print(string_list)

def is_empty(any_structure):
    if any_structure:
        print('Structure is not empty.')
        return False
    else:
        print('Structure is empty.')
        return True

def num_cols(sheet):
    """ Find the number of columns in a sheet"""
    cols = [len(sheet[row]) for row in range(len(sheet))]
    return max(cols, default=0)

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    vals = set([row[col] for row in rows])
    unique = sorted(list(vals))
    return unique

def is_numeric(value):
    """Test if a value is numeric"""
    return isinstance(value, numbers.Number)

def class_counts(rows, col):
    """Counts the number of each type in the column. Categorical data only"""
    counts = {}
    for row in rows:
        label = row[col]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def dec_counts(rows, col):
    """Provides proportions for each category of a column"""
    counts = class_counts(rows, col)
    total = sum(counts.values()) * 1.0
    prop = {}
    for lbl in counts.keys():
        prop[lbl] = counts[lbl] / total
    return prop

def flatten_generator(nested_list):
    """Generator to flatten an arbitrarily nested list. To extract
    results use list(flatten(x))"""
    for i in nested_list:
        if isinstance(i, (list, tuple)):
            for j in flatten_generator(i):
                yield j
        else:
            yield i

def flatten(nested_list):
    """Converts the flatten_generator func results into a list"""
    return list(flatten_generator(nested_list))

def get_key(val, my_dict):
    """function to return key for any value"""
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def sortdict(my_dict, descending=True, sortbykeys=False):
    """Sorts dictionary by key or values"""
    if sortbykeys:
        keys = sorted(my_dict.keys(), reverse=descending)
        return {k:my_dict[k] for k in keys}

    values = sorted(my_dict.values(), reverse=descending)
    return {get_key(v, my_dict):v for v in values}

def split_train_test(data, tst_prop=0.5):
    """Split a data set on its rows into test and train data sets."""
    random.shuffle(data)
    train_size = int(len(data) - tst_prop*len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data
