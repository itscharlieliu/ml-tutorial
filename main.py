from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix


def main():
    url = "./iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)

    print("[Dataset shape]\n{}\n".format(dataset.shape))
    print("[Dataset preview]\n{}\n".format(dataset.head(10)))
    print("[Dataset summary]\n{}\n".format(dataset.describe()))
    print("[Class distribution]\n{}\n".format(dataset.groupby("class").size()))

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    dataset.hist()
    scatter_matrix(dataset)
    pyplot.show()


if __name__ == "__main__":
    main()
