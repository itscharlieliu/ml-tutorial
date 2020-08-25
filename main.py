from matplotlib import pyplot
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def main():
    url = "./iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)

    # print("[Dataset shape]\n{}\n".format(dataset.shape))
    # print("[Dataset preview]\n{}\n".format(dataset.head(10)))
    # print("[Dataset summary]\n{}\n".format(dataset.describe()))
    # print("[Class distribution]\n{}\n".format(dataset.groupby("class").size()))
    #
    # # box and whisker plots
    # dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    # dataset.hist()
    # scatter_matrix(dataset)
    # pyplot.show()

    # Split-out validation dataset
    array = dataset.values

    x = array[:, 0:4]
    y = array[:, 4]
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    pyplot.boxplot(results, labels=names)
    pyplot.title("Algorithm Comparison")
    pyplot.show()


if __name__ == "__main__":
    main()
