import numpy as np

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        # The step size for Gradient Descent
        self.learning_rate = learning_rate
        # How many times we pass through the data to optimize weights
        self.num_iterations = num_iterations
        
        # W represents feature weights (how important is 'amount' vs 'step'?)
        self.weights = None
        # B represents the bias (base probability of fraud)
        self.bias = None

    def _sigmoid(self, z):
        """
        The core activation function.
        It maps any real number into a probability between 0 and 1.
        Formula: 1 / (1 + e^-z)
        """
        # np.clip prevents mathematical overflow on very massive numbers 
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Gradient Descent Optimization
        This is where the AI actually 'learns'.
        """
        num_samples, num_features = X.shape
        
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 2. Gradient Descent Loop
        for i in range(self.num_iterations):
            
            # Step A: Forward Pass (Predict)
            # z = (X * Weights) + Bias
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Step B: Calculate Gradients (How wrong were we?)
            # dw = Derivative with respect to weights
            # db = Derivative with respect to bias
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Step C: Update Weights (Move in the direction of 'less wrong')
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # (Optional) Print loss every 100 iterations to watch it learn
            if i % 100 == 0:
                loss = self._binary_cross_entropy(y, y_predicted)
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """Returns the exact percentage probability of Fraud"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
        
    def predict(self, X, threshold=0.5):
        """Returns a hard 1 (Fraud) or 0 (Legitimate)"""
        probabilities = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probabilities]

    def _binary_cross_entropy(self, y_true, y_pred):
        """The Log-Loss function that calculates how penalized the model is for being wrong"""
        # Add a tiny epsilon to prevent log(0) which throws an infinity error
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
        
if __name__ == "__main__":
    print("Initializing Fraud Data (Fake Sample for Speed)...")
    
    # 5 Sample Transactions, 3 features each: [Amount, Sender_Balance_Wiped, Destination_Zero]
    X_train = np.array([
        [15.00, 0, 0],   # Normal payment
        [45.00, 0, 0],   # Normal payment
        [9500., 1, 1],   # FRAUD: Huge amount, balance wiped, destination empty
        [2500., 0, 0],   # Normal big transfer
        [150000, 1, 1],  # FRAUD: Massive Account Takeover!
    ])
    
    # 0 = Legitimate, 1 = Fraud
    y_train = np.array([0, 0, 1, 0, 1])
    
    print("\nTraining Custom Logistic Regression Model from Scratch...")
    model = LogisticRegressionFromScratch(learning_rate=0.01, num_iterations=500)
    model.fit(X_train, y_train)
    
    print("\nTesting a new transaction: $15,000 sent to an empty destination, wiping sender balance.")
    X_test = np.array([[15000.0, 1, 1]])
    
    proba = model.predict_proba(X_test)[0]
    prediction = model.predict(X_test)[0]
    
    print(f"\nResult: {proba*100:.2f}% Probability of Fraud.")
    print(f"Verdict: {'FRAUD 🚨' if prediction == 1 else 'LEGITIMATE ✅'}")

