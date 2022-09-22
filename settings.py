from sklearn.tree import DecisionTreeClassifier

from Code.TreeStrains import TreeStrainsClassifier

def getDataset():
    datasets = [
        './Datasets/avila.csv',
        './Datasets/banknote.csv',
        './Datasets/cancerwisconsin.csv',
        './Datasets/car.csv',
        './Datasets/cargo.csv',
        './Datasets/credit.csv',
        './Datasets/crowd.csv',
        './Datasets/diabetes.csv',
        './Datasets/digits.csv',
        './Datasets/frog-family.csv',
        './Datasets/frog-genus.csv',
        './Datasets/frog-species.csv',
        './Datasets/hcv.csv',
        './Datasets/htru.csv',
        './Datasets/ionosfera.csv',
        './Datasets/iranian.csv',
        './Datasets/iris.csv',
        './Datasets/mice.csv',
        './Datasets/mushroom.csv',
        './Datasets/obesity.csv',
        './Datasets/occupancy.csv',
        './Datasets/pen.csv',
        './Datasets/qualitywine.csv',
        './Datasets/robot.csv',
        './Datasets/sensorless.csv',
        './Datasets/shill.csv',
        './Datasets/sonar.csv',
        './Datasets/taiwan.csv',
        './Datasets/thyroid.csv',
        './Datasets/vowel.csv',
        './Datasets/wifi.csv',
        './Datasets/wine.csv',
        './Datasets/20newsgroups.csv',
        './Datasets/data0.csv',
        './Datasets/data5.csv',
        './Datasets/data10.csv',
        './Datasets/data25.csv',
        './Datasets/data50.csv',
        './Datasets/myocardial.csv',
        './Datasets/micromass.csv'
    ]

    return datasets


def getClassifier():
    model = TreeStrainsClassifier(n_estimators=50, base_estimator=DecisionTreeClassifier(),
                        similarity_metric='laplacian', autofill=True)
    return model

