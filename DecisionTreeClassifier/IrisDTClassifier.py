from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
import seaborn as sns
%matplotlib inline
sns.set(style="ticks" , color_codes=True)
g = sns.pairplot(sns.load_dataset('iris'), hue = 'species')
g.savefig("pairpltiris.png")
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data,iris.target)
with open("iris_classifier.dot", "w") as ic:
    ic = tree.export_graphviz(clf,out_file = ic)
#dot -Tpdf iris_classifier.dot -o iris_clf.pdf