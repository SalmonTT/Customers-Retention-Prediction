from decisionTree import *
from deepLearning import *
from dataPreprocessing import getTrainingData




def main():
    # ----- Part 1: Data Pre-processing ----- #
    train = getTrainingData("train.csv")
    print(train)
    # ----- Part 2: Model Training ----- #
    # useDecisionTree(train)
    useKeras(train)

    # hi testing
if __name__ == "__main__":
    main()
