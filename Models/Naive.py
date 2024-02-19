class naive:
    def predict(self,X_test):
        prediction_naive = np.zeros(len(X_test))  # Initialize a 1D array for predictions
        for i, distance in enumerate(X_test['distances']):
            if distance < 0.7:
                prediction_naive[i] = 1
            else:
                prediction_naive[i] = 0
        return prediction_naive
