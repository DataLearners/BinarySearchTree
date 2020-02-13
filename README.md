

## Setting up Visual Studio Code

https://stackoverflow.com/questions/1899436/pylint-unable-to-import-error-how-to-set-pythonpath

pylint rc
https://gist.github.com/xen/6334976


## Multi Processor Strategy

Map Reduce at the tree level.

Each tree gets its own processor.
Trees will only know about each other in the reduce part of the map reduce.


## Domain Defined

main
    tree = TreeBuilder(TreeArgs)
    diagnostics = TreeDiagnostics(DiagArgs, tree, tree_args)

Tree - data class its the final output after running all the calculations. For the most part it is imutable.
TreeArgs - The settings to build a tree
TreeBuilder - Methods to generate a tree
TreeDiagnostics - checks to see if the tree was built correctly

Leaf - 
Decisions
Questions

Forest - final output mostly imutable
Forest - 



# Things to research

How to import libraries into Zeplin or will we just have one big file?
