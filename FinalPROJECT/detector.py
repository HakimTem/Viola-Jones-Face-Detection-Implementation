from image_util import IntegralImage as ii
from haar_feature import HaarFeature as haar
from classifier import ClassifierStage


class ViolaJonesDetector:

    def __init__(self):
        self.classifier_cascade = []

    def classify(self, image):
        for stage in self.classifier_cascade:
            if stage(image) == 0:
                return 0
        return 1

    def evaluate(self, image_set):
        D, F = 0, 0
        num_pos, num_neg = 0, 0
        for image, label in image_set:
            guess = self.classify(image)

            if guess == 1 and label == 1:
                D += 1
            if guess == 1 and label == 0:
                F += 1

            if label == 1:
                num_pos += 1
            else:
                num_neg += 1

        print(f"False Positives: {F}")
        print(f"Number of Negative Examples: {num_neg}")
        print(f"Detecttion Rate: {D / num_pos}")
        print(f"False Positive Rate: {F / num_neg}")
        return D / num_pos, F / num_neg

    def adjust_detection_threshold(self, validation_set, D_target, clf_idx):
        D, F = self.evaluate(validation_set)
        while D < D_target:
            self.classifier_cascade[clf_idx].strong_classifier.gamma -= 0.01
            D, F = self.evaluate(validation_set)
            print(f"Stage Number: {clf_idx + 1}")
            print(f"Number of Stages: {len(self.classifier_cascade)}")
            print(f"Gamma: {self.classifier_cascade[clf_idx].strong_classifier.gamma}")
        return D, F

    def record_false_positives(self, N):
        FP = []
        for image, label in N:
            if self.classify(image) == 1:
                FP += [(image, label)]
        return FP

    def train(self, training_dataset, f, d, F_target):
        P = []
        N = []
        training = ii.integrate_dataset(training_dataset)

        for ii_set in training:
            if ii_set[1] == 1:
                P.append(ii_set)
            else:
                N.append(ii_set)

        print(f"Num Positive Examples: {len(P)}")
        print(f"Num Negative Examples: {len(N)}")

        F, D = [1.0], [1.0]
        i = 0

        feature_max_size = (19, 19)
        features = haar.build_feature_space(feature_max_size)
        print(f"Number of Haar Features: {len(features)}")

        while F[i] > F_target:
            X, Y = haar.evaluate_features(features, P + N)

            i += 1
            F.append(F[i - 1])
            D.append(1.0)

            n = 1
            stage = ClassifierStage()
            training_weights = stage.adaboost_train(P + N, features, X, Y, len(P), len(N), 1)
            self.classifier_cascade.append(None)
            while F[i] > f * F[i - 1]:
                n += i
                self.classifier_cascade.pop()
                training_weights = stage.adaboost_train(P + N, features, X, Y, len(P), len(N), i, training_weights)
                self.classifier_cascade.append(stage)
                D[i], F[i] = self.adjust_detection_threshold(training, d * D[i - 1], i - 1)

            print("Found stage")
            print(f"False Positive Rate: {F[i]}")
            print(f"Detection Rate: {D[i]}")

            if F[i] > F_target:
                N = self.record_false_positives(N)

        print("IM done")
