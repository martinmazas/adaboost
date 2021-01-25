from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)


def convert_columns_into_integers_values(dataframe):
    # Convert string dataframe values in integers
    new_dataframe = dataframe.copy()
    new_dataframe.pclass = dataframe.pclass.map({'crew': 0, '1st': 1, '2nd': 2, '3rd': 3})
    new_dataframe.gender = dataframe.gender.map({'male': 1, 'female': 0})
    new_dataframe.age = dataframe.age.map({'adult': 1, 'child': 0})
    new_dataframe.survived = dataframe.survived.map({'yes': 1, 'no': 0})
    return new_dataframe


def adaboost():
    # Save original data into data frames
    titanic = pd.read_csv('titanikData.csv')
    header = titanic.columns.values
    titanic_test = pd.read_csv("titanikTest.csv", names=header)

    titanic_size = len(titanic)

    titanic_training_data = convert_columns_into_integers_values(titanic)
    titanic_test_data = convert_columns_into_integers_values(titanic_test)

    for i in range(len(titanic)):
        # for each example in data set weight 1/n
        titanic_training_data['weight'] = 1/titanic_size

    hypothesis = []
    alpha_list = []

    titanic_test_input = titanic_test_data.drop('survived', axis=1)

    features = titanic.columns.drop('survived')
    x = titanic_training_data[features]
    y = titanic_training_data.survived
    for i in range(3):
        clf = tree.DecisionTreeClassifier()
        clf.max_depth = 1
        clf.criterion = 'entropy'
        clf = clf.fit(x, y)

        # Add hypothesis into set of hypothesis
        hypothesis.append(clf)
        titanic_training_data['prediction'] = clf.predict(x)

        # If prediction is incorrect put 1 else 0
        titanic_training_data['incorrect_prediction'] = np.where(titanic_training_data['prediction'] !=
                                                                 titanic_training_data['survived'], 1, 0)

        # Calculate error rate
        error_rate = np.sum(titanic_training_data.weight * titanic_training_data.incorrect_prediction)

        if error_rate > 0.5:
            continue

        # Calculate beta and alpha. Each alpha will be used later
        beta = error_rate/(1-error_rate)
        alpha = 0.5 * np.log(1/beta)
        alpha_list.append(alpha)

        # If prediction is True weight*beta else weight*alpha
        titanic_training_data['weight'] = np.where(titanic_training_data['incorrect_prediction'] == 0,
                                                   titanic_training_data['weight']*beta,
                                                   titanic_training_data['weight']*alpha)
        # Normalize the weights
        weight_sum = sum(titanic_training_data['weight'])
        titanic_training_data['weight'] = titanic_training_data['weight']/weight_sum

    final_prediction = []
    for (j, i) in enumerate(hypothesis):
        final_prediction.append(alpha_list[j] * i.predict(titanic_test_input))

    predictions = [sum(x) for x in zip(*np.array(final_prediction))]

    for i in range(len(predictions)):
        predictions[i] = 'yes' if predictions[i] == 0 else 'no'

    predict_decision = pd.DataFrame(predictions, columns=['prediction'])
    titanic_test_final_data = pd.concat([titanic_test, predict_decision], axis=1)

    print(titanic_test_final_data)
    print("Successfully predicted data with accuracy " + "= "
          + str(accuracy_score(titanic_test_final_data.survived, titanic_test_final_data.prediction) * 100) + ' %')


def main():
    adaboost()


if __name__ == "__main__":
    main()