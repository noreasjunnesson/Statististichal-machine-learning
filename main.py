import csv
from model_a import ModelA
    from qda import qda


def load_data(filename):
    # Load data from CSV file
    # Return training and testing data
    return train_data, test_data

def main():
    # Load data
    train_data, test_data = load_data("data.csv")

    # Instantiate models
    model_a = ModelA()
    qda = qda()

    # Train models
    trained_model_a = model_a.train(train_data)
    trained_qda =qda.train(train_data)

    # Evaluate models
    results_a = evaluate_model(trained_model_a, test_data)
    results_qda = evaluate_model(trained_qda, test_data)

    # Compare results or perform further analysis
    # For example:
    print("Results from Model A:", results_a)
    print("Results from QDA:", results_qda)

if __name__ == "__main__":
    main()
