from decisionTree import *
from dataPreprocessing import getTrainingData




def main():
    # ----- Part 1: Data Pre-processing ----- #
    train = getTrainingData("train.csv")
    # ----- Part 2: Model Training ----- #
    useDecisionTree(train)



if __name__ == "__main__":
    main()
