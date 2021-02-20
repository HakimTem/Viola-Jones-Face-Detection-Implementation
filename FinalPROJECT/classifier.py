import numpy as np
import math


# "Weak" classifier class used for determining whether image is human
# face. Uses feature and optimized parameters
class WeakClassifier:

    # Constructor for Weak Classifier
    #   feature - Haar Feature to use on image to obtain feature value
    #
    #   threshold - the value which determines whether feature value
    #               corresponds to face or non-face
    #
    #   polarity - value used for determining the direction of inequality
    def __init__(self, feature, threshold, polarity):
        self.f = feature
        self.theta = threshold
        self.p = polarity

    # Method used for determining wheter image is face or non-face using
    # given feature value or calculated feature value. If feature value is within threshold it is declared a face
    #
    # param : x - image to evaluate feature on
    #         f_x - feature value if already obtained of image
    def __call__(self, x, f_x=None):
        if f_x is None:
            f_x = self.f(x)

        if self.p == 1 and f_x < self.theta:
            return 1
        elif self.p == -1 and f_x > self.theta:
            return 1
        return 0


# Strong Classifier used for determining whether given image is
# human face. Uses linear combination of weak classifiers
class StrongClassifier:

    # Constructor for Strong classifier
    #   weak_learners - a list of chosen weak learners
    #   alpha_vector - a list of weights associated with weak learnears
    #   gamma - total threshold for determining whether image is face
    def __init__(self, weak_learners, alpha_vector):
        self.weak_classifiers = weak_learners
        self.alphas = alpha_vector
        self.gamma = 0.5 * sum(self.alphas)

    # Method used for determining whether image is face or non-face
    # using combination of weak learners
    #
    # param : x - desired image to be classified
    def __call__(self, x):
        weighted_sum = 0
        for classifier, alpha in zip(self.weak_classifiers, self.alphas):
            weighted_sum += alpha * classifier(x)
            if weighted_sum >= self.gamma:
                return 1
        return 0

    # Method used for appending a new weak learner along with an accosiated
    # weight. Recalculate the total threshold
    #
    # param : h - the weak classifier to append
    #         alpha - weight of the weak classifier
    def append_voter(self, h, alpha):
        self.weak_classifiers.append(h)
        self.alphas.append(alpha)

        # variable used for the total threshold of strong classifier by linear combination of alphas
        self.gamma += 0.5 * alpha


# Class used for independent stage of face detection. Trains its own
# strong classifier with training and validation set
class ClassifierStage:

    # Constructor for Strong classifier, simply declares a variable for storage of strong classifier
    def __init__(self):
        self.strong_classifier = None

    # Method used for appending a new weak learner along with an accosiated
    # weight. Recalculate the total threshold
    #
    # param : x - The image to classify for face detection
    def __call__(self, x):
        return self.strong_classifier(x)

    # Method used for creating optimized classifiers based on specific
    # features associated with custom thresholds and polarity values.
    # (Uses training set along with image weights)
    #
    # param : features - a list of features to optimize and form weak classifiers over
    #         X - 2D rray that contains every feature value evaluated over every image set
    #         Y - list of labels associated for each image distinguishing if image is face or non-face
    #         weights - the "weight" for each image that highlights the importance of a certain image in training
    def optimize_classifiers(self, features, X, Y, weights):
        # Container for optimized classifiers
        optimal_classifiers = []

        # Record the total positive weights and the total negative weights
        total_pos_weights, total_neg_weights = 0, 0
        for weight, y in zip(weights, Y):
            if y == 1:
                total_pos_weights += weight
            else:
                total_neg_weights += weight

        print(f"Total Positive Weights: {total_pos_weights}")
        print(f"Total Negative Weights: {total_neg_weights}")

        #
        for feature_idx in range(len(features)):
            threshold, polarity, best_error = 0, 0, float('inf')
            seen_pos_weights, seen_neg_weights = 0, 0
            pos_seen, neg_seen = 0, 0

            sorted_feature_values = sorted(zip(weights, Y, X[feature_idx]), key=lambda data: data[2])
            # print(f"Sorted Feature Values: {sorted_feature_values}")

            for img_vector in sorted_feature_values:
                if img_vector[1] == 1:
                    pos_seen += 1
                    seen_pos_weights += img_vector[0]
                else:
                    neg_seen += 1
                    seen_neg_weights += img_vector[0]

                error = min(seen_pos_weights + (total_neg_weights - seen_neg_weights),
                            seen_neg_weights + (total_pos_weights - seen_pos_weights))

                if error < best_error:
                    best_error = error
                    threshold = img_vector[2]
                    polarity = 1 if pos_seen > neg_seen else -1
            optimal_classifiers.append(WeakClassifier(features[feature_idx], threshold, polarity))
        return optimal_classifiers

    # Method used for selecting the best classifier from a list of
    # optimized classifiers by testing their accuracy on weighted
    # training set
    def select_classifier(self, classifiers, X, Y, weights):
        chosen_clf, accuracy, best_error = None, None, float('inf')
        for clf_idx in range(len(classifiers)):
            guesses = [classifiers[clf_idx](None, f_x) for f_x in X[clf_idx]]
            correctness = np.absolute(np.subtract(guesses, Y))

            clf_error = np.sum(np.multiply(weights, correctness))
            if clf_error < best_error:
                best_error = clf_error
                accuracy = correctness
                chosen_clf = classifiers[clf_idx]

        print(f"Classifier Error Rate: {sum(accuracy) / len(Y)}")
        return chosen_clf, best_error, accuracy

    # Method used for general adaboost training method to produce the best
    # strong classifier for a training set
    def adaboost_train(self, training_set, features, X, Y, num_pos, num_neg, T, weights=None):

        if weights is None:
            weights = np.zeros(len(Y))
            for i, _ in enumerate(weights):
                if Y[i] == 1:
                    weights[i] = 1 / (2 * num_pos)
                else:
                    weights[i] = 1 / (2 * num_neg)

        for t in range(0, T):
            print(f"Classifier Round: {t}")

            weights = np.divide(weights, np.sum(weights))

            classifiers = self.optimize_classifiers(features, X, Y, weights)

            print("Found optimal classifiers")

            best_classifier, best_error, best_accuracy = self.select_classifier(classifiers, X, Y, weights)

            beta = best_error / (1.0 - best_error)
            if best_error <= 0:
                alpha = 2.0
            else:
                alpha = math.log10(1.0 / beta)

            if self.strong_classifier is None:
                self.strong_classifier = StrongClassifier([best_classifier], [alpha])
            else:
                self.strong_classifier.append_voter(best_classifier, alpha)

            if best_error <= 0:
                break

            best_feature = best_classifier.f
            print(f"Best Error: {best_error}, Alpha: {alpha}, Classifier: {best_feature.featureType}")

            best_feature_idx = features.index(best_feature)
            print(f"Best Feature Index: {best_feature_idx}")
            X[best_feature_idx] = np.zeros(len(training_set))

            for i in range(0, len(weights)):
                weights[i] = weights[i] * (beta ** (1 - best_accuracy[i]))
        return weights
