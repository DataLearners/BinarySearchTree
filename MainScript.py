import random
import matplotlib.pyplot as plt
from datetime import datetime
import startup
startup.add_library('data_science_poc\\library')
import bst
random.seed(10)

# ##########Random Test Data#################
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
fruitdf = bst.ModelDF(bst.ListDF(data, header), ycol=0)
# #############Build a Tree###############################
treeparams = bst.TreeParams(mingain=0, minrows=1, maxheight=15)
configs = bst.Settings(fruitdf, treeparams)
node = bst.Node(fruitdf.train, configs, 1)
column = bst.Column(fruitdf.data, 1, fruitdf.header)
ginitree = bst.Tree(fruitdf)

# ###########Build a Forest###############################
ginifrst = bst.Forest(fruitdf)

fruitdf = bst.ModelDF(bst.ListDF(data, header), ycol=1)
regtree = bst.Tree(fruitdf)
regfrst = bst.Forest(fruitdf, nfeatures=1, ntrees=25)

# #############Build a Bunch of Forests##################
print('\nClassification Out of Bag Error')
fruitdf = bst.ModelDF(bst.ListDF(data, header), ycol=0)
trees, ooberr = [], []
for i in range(2, 100, 20):
    ginifrst = bst.Forest(fruitdf, nfeatures=1, ntrees=i)
    trees.append(i)
    ooberr.append(ginifrst.oob_err)
    print(ginifrst.oob_err)
plt.plot(trees, ooberr)
plt.ylabel('Out of Bag Error')
plt.show()

print('\nRegression Out of Bag Error')
fruitdf = bst.ModelDF(bst.ListDF(data, header), ycol=1)
trees, ooberr = [], []
for i in range(2, 100, 12):
    regfrst = bst.Forest(fruitdf, nfeatures=1, ntrees=i)
    trees.append(i)
    ooberr.append(regfrst.oob_err)
    print(regfrst.oob_err)
plt.plot(trees, ooberr)
plt.ylabel('Out of Bag Error')
plt.show()
#
## #######################External Data##########################
#path = startup.find_root('\\Documents\\Python Scripts\\Datasets')
#files = startup.Load(path)
## #Diabetes
#data, header = files(1)
#yes_no = lambda x: {1: 'Yes', 0: 'No'}[x]
#df = bst.ListDF(data, header)
#df.transform_col(yes_no, 'Outcome')
#
#diab = bst.ModelDF(df, ycol=8, tst_prop=0.5)
#diabtree = bst.Tree(diab)
#diabtree.nodesummary.export('TreeSummary.csv')
#print(diabtree.fitsummary)
#
## #GFM
#data, header = files(0)
#header[0] = 'Orgid'
#df = bst.ListDF(data, header)
#df.del_cols(colnames=['Augmentation_Parent'])  # no variance
#
#types = lambda i: list(set(map(type, df.unique_vals(i))))
#multi_dtypes = [x for i, x in enumerate(df.header) if len(types(i)) > 1]
#df.del_cols(colnames=multi_dtypes)  # creates errors
#
#too_high_var = [df.header[i] for i, x in enumerate(df.levels) if x > 1000]
#df.del_cols(colnames=too_high_var)
#
#def date_to_num(datestring):
#    """Convert datestring to number. Earliest possible date is 1/2/1970"""
#    year = datetime.fromisoformat(datestring).year
#    if year <= 1970:
#        return datetime.fromisoformat('1970-01-02 00:00:00.000').timestamp()
#    return datetime.fromisoformat(datestring).timestamp()
#
#date_cols = [x for x in df.header if 'date' in x.lower()]
#[df.transform_col(date_to_num, x) for x in date_cols]
#
#org_type_dict = {1: 'DeployableUnit', 2: 'CrewPlatform', 4: 'DoctrinalOrg'}
#org_type = lambda n: org_type_dict[n]
#df.transform_col(org_type, 'OrgEntityTypeId')
#
#bst.Tree.counter = 0
#gfm = bst.ModelDF(df, ycol=0, tst_prop=0.3)
#gfmtree = bst.Tree(gfm, maxheight=20)
#print(gfmtree.fitsummary)
#gfmtree.nodesummary.export('NodeSummary.csv')
#gfmtree.leafclusters.export('TreeClusters.csv')
#gfmtree.leafmatrix.export('LeafMatrix.csv')
#print(gfmtree.featureimportance)
