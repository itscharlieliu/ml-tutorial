from pandas import read_csv


def main():
    url = "./iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)

    print("Dataset shape: {}".format(dataset.shape))


if __name__ == "__main__":
    main()
