import numpy as np
from collections import Counter

# ==============================================================================
# 1. THE DECISION TREE COMPONENT
# ==============================================================================

class Node:
    """A single node inside a Decision Tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       # The index of the feature we split on (e.g. 'amount')
        self.threshold = threshold   # The numeric threshold we split at (e.g. 50000)
        self.left = left             # Left child Node
        self.right = right           # Right child Node
        self.value = value           # If it's a leaf node, this is the predicted class (0 or 1)

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    """A single Decision Tree built using Information Gain (Entropy)"""
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        # Determine how many features to look at per split
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria (Max depth reached, pure node, or too few samples)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Randomly select a subset of features to evaluate (The 'Random' in Random Forest)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Greedily search for the best feature and threshold to split the data
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Split the data
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        
        # Recursively grow the left and right branches
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        # Look through all selected features
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            # Test every possible threshold for this feature
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # 1. Calculate the base Entropy of the parent node
        parent_entropy = self._entropy(y)

        # 2. Split the node
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # 3. Calculate the weighted Entropy of the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Information Gain is the reduction in Entropy (Chaos)
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        # Calculate chaos/impurity using the Shannon Entropy formula: -Sum(p * log2(p))
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        # Go left or right depending on the feature threshold
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# ==============================================================================
# 2. THE RANDOM FOREST ENSEMBLE
# ==============================================================================

class RandomForestFromScratch:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees                       # Number of trees in the forest
        self.max_depth = max_depth                   # How deep each tree can grow
        self.min_samples_split = min_samples_split   # Minimum samples required to split a node
        self.n_features = n_features                 # Number of features to consider per split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            # 1. Bootstrap: Create a random sub-sample of the dataset (with replacement)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            
            # 2. Build and train a new unique tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            print(f"   🌳 Tree {i+1}/{self.n_trees} grown successfully.")

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        # Ramdomly pick indices with replacement
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        """Aggregate votes from all trees (Majority Voting)"""
        # Get predictions from every tree: Array shape (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Swap axes to (n_samples, n_trees) so we can vote per transaction
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # Soft voting: Find the most common prediction across the forest
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions

    def predict_proba(self, X):
        """Calculates the percentage of trees that voted 'Fraud' vs 'Legit'"""
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        probas = []
        for preds in tree_preds:
            # Ratio of trees that voted '1'
            fraud_prob = np.sum(preds) / self.n_trees
            # Return [Legit Prob, Fraud Prob] to match Scikit-Learn perfectly
            probas.append([1 - fraud_prob, fraud_prob])
            
        return np.array(probas)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


# ==========================================
# DEMONSTRATION RUN
# ==========================================
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
    
    print("\nTraining Custom Random Forest Model from Scratch...")
    # NOTE: Training 10 trees recursively in raw Python is heavy!
    forest = RandomForestFromScratch(n_trees=10, max_depth=3)
    forest.fit(X_train, y_train)
    
    print("\nTesting a new transaction: $15,000 sent to an empty destination, wiping sender balance.")
    X_test = np.array([[15000.0, 1, 1]])
    
    proba = forest.predict_proba(X_test)[0][1]
    prediction = forest.predict(X_test)[0]
    
    print(f"\nResult: {proba*100:.2f}% of our Trees voted that this is Fraud.")
    print(f"Verdict: {'FRAUD 🚨' if prediction == 1 else 'LEGITIMATE ✅'}")

