import hk00570_functions as fct
import pandas as pd

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 999)

# GENRE
print('GENRE')
genre_class = fct.HKClassifierClass('genre')

# explore data set
genre_class.explore_data()

# pre-process data set
genre_class.pre_process_data()

# SOM classification
print('GENRE - SOM Classification')
# explore a variety of values for learning rate, sigma and neighborhood function
genre_class.explore_som_classification_parameters()

genre_class.execute_som_model(150, 5000, 15, 0.01, 'gaussian')

genre_class.som_10_fold_cross_validation(150, 5000)


# Decision Tree Classifier
print('GENRE - Decision Tree Classification')

tree = genre_class.execute_tree_model()

genre_class.explore_tree_pruning()

genre_class.tree_10_fold_cross_validation(tree)


genre_class.plot_ROC_curve()

## POPULARITY
print(' ')
print('POPULARITY')
poularity_class = fct.HKClassifierClass('popularity')

# explore data set
poularity_class.explore_data()

# pre-process data set
poularity_class.pre_process_data()

# SOM classification
print('POPULARITY - SOM Classification')
poularity_class.explore_som_classification_parameters()

poularity_class.execute_som_model(150, 5000, 5, 0.01, 'gaussian')

poularity_class.som_10_fold_cross_validation(150, 5000)


# Decision Tree Classifier
print('POPULARITY - Decision Tree Classification')
tree = poularity_class.execute_tree_model()

poularity_class.explore_tree_pruning()

poularity_class.tree_10_fold_cross_validation(tree)


poularity_class.plot_ROC_curve()
