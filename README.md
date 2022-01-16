
# This prepository contains basic `Decision Tree` implementation 


This implementation uses the following loss function for features selection:

$Q(R) = -\frac {|R_{\ell}|}{|R|}H(R_\ell) -\frac {|R_r|}{|R|}H(R_r) $

<img src="https://render.githubusercontent.com/render/math?math=Q(R) = -\frac {|R_{\ell}|}{|R|}H(R_\ell) -\frac {|R_r|}{|R|}H(R_r)">

where $R$ — set of objects, in a tree, $R_{\ell}$ and $R_r$ — objects, in a left and right sub-trees,
$H(R)$ — impurity criterion(Gini or Entropy).

In this implementation, a naive partitioning algorithm is used for categorical features: we are trying to find one value, the partition by which will most strongly improve the quality functional at a given step. In other words, objects with a specific attribute value are sent to the left subtree, the rest - to the right. Please note that this is far from the optimal way to take into account categorical features. For example, one could create a separate subtree for each value of a categorical attribute, or use more complex approaches.You can read more about this in the [lecture notes](https://github.com/esokolov/ml-course-hse/blob/master/2019-fall/lecture-notes/lecture07-trees.pdf) on machine learning on PMI (section "Accounting for categorical features").  

ToDo:

- Rewrite the implementation without using for-loops. 

