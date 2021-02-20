import enum
from tqdm import tqdm, trange
from multiprocessing import Process, Manager


# Method for evaluating features over an image set
# METHOD EDITS A LIST DOES NOT CREATE ONE!
def eval_features(ii_set, features, X):
    for f in tqdm(features):
        X.extend([list(map(lambda ii: f(ii[0]), ii_set))])


# Enum class used for distinguishing between different feature types
class FeatureType(enum.Enum):
    two_horizontal_feature = 1
    two_vertical_feature = 2
    three_horizontal_feature = 3
    three_vertical_feature = 4
    four_diagonal_feature = 5


# Class used for calculating different haar features on images
class HaarFeature:

    # Method used for building all the possible haar features for a given
    # image
    #
    # params : image_shape - tuple of height and width
    @staticmethod
    def build_feature_space(image_shape):
        height, width = image_shape
        features = []

        # Loop through all possible size and coordinate combinations for specific haar features
        for w in trange(1, width + 1):
            for h in range(1, height + 1):
                x = 0
                while x + w <= width:
                    y = 0
                    while y + h <= height:
                        if h % 2 == 0:
                            features.append(HaarFeature(FeatureType.two_horizontal_feature, x, y, w, h))
                        if w % 2 == 0:
                            features.append(HaarFeature(FeatureType.two_vertical_feature, x, y, w, h))
                        if h % 3 == 0:
                            features.append(HaarFeature(FeatureType.three_horizontal_feature, x, y, w, h))
                        if w % 3 == 0:
                            features.append(HaarFeature(FeatureType.three_vertical_feature, x, y, w, h))
                        if w % 2 == 0 and h % 2 == 0:
                            features.append(HaarFeature(FeatureType.four_diagonal_feature, x, y, w, h))
                        y += 1
                    x += 1
        return features

    # Method used for evaluating features on an image set
    #
    # param : f_vector - list of haar features to use on image set
    #         ii_set - integral image set composed of numpy array of form ((19,19), 1)
    @staticmethod
    def evaluate_features(f_vector, ii_set, X=None):
        X = []
        Y = list(map(lambda y: y[1], ii_set))

        # chunk_size = int(len(f_vector) / 4)
        # print(f"Chunk Size: {chunk_size}")
        #
        # splitted_features = [f_vector[i:(i + chunk_size)] for i in range(0, len(f_vector), chunk_size)]
        #
        # import time
        # start = time.perf_counter()

        eval_features(ii_set, f_vector, X)
        # with Manager() as manager:
        #     X = manager.list()
        #
        #     # Testing the usability of multiprocessing for running parellel programs
        #     processes = []
        #     for i in range(4):
        #         p = Process(target=eval_features, args=[ii_set, splitted_features[i], X])
        #         p.start()
        #         processes.append(p)
        #
        #     for p in processes:
        #         p.join()
        #
        #     X = list(X)
        # end = time.perf_counter()
        # print(f"Time Taken: {end - start}")
        # print(X)
        return X, Y

    # Constructor for Haar Feature
    #  type - FeatureType for how the feature is evaluated on an image
    #  x - X-coordinate of the feature on a given image
    #  y - y-coordinate of the feature on a given image
    #  width - width of the feature (must be multiple of featureType)
    #  height - height of the feature (must be multiple of featureType)
    def __init__(self, type, x, y, width, height):
        self.featureType = type
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    # Method used for evaluating a specific haar feature on a given image
    # param : ii - integral image of form (19,19) to evaluate Haar Feature on
    def __call__(self, ii):
        # Rescale size and coordinates of haar feature based on image subwindow
        w = int(self.width * ((ii.shape[0] - 1) / 19))
        h = int(self.height * ((ii.shape[0] - 1) / 19))
        y, x = self.y, self.x

        # Based on the feature type, compute the difference in pixel areas around specific regions using
        # Integral Image Array Lookup

        if self.featureType == FeatureType.two_horizontal_feature:
            f_h = int(h / 2)
            A = ii[y, x]
            B = ii[y, x + w]
            C = ii[y + f_h, x]
            D = ii[y + f_h, x + w]
            E = ii[y + h, x]
            F = ii[y + h, x + w]
            return (2 * D + A + E) - (B + 2 * C + F)

        elif self.featureType == FeatureType.two_vertical_feature:
            f_w = int(w / 2)
            A = ii[y, x]
            B = ii[y, x + f_w]
            C = ii[y, x + w]
            D = ii[y + h, x]
            E = ii[y + h, x + f_w]
            F = ii[y + h, x + w]
            return (D + 2 * B + F) - (C + A + 2 * E)

        elif self.featureType == FeatureType.three_horizontal_feature:
            f_h = int(h / 3)
            A = ii[y, x]
            B = ii[y, x + w]
            C = ii[y + f_h, x]
            D = ii[y + f_h, x + w]
            E = ii[y + 2 * f_h, x]
            F = ii[y + 2 * f_h, x + w]
            G = ii[y + h, x]
            H = ii[y + h, x + w]
            return (2 * F + 2 * C + B + G) - (2 * E + 2 * D + A + H)

        elif self.featureType == FeatureType.three_vertical_feature:
            f_w = int(w / 3)
            A = ii[y, x]
            B = ii[y, x + f_w]
            C = ii[y, x + 2 * f_w]
            D = ii[y, x + w]
            E = ii[y + h, x]
            F = ii[y + h, x + f_w]
            G = ii[y + h, x + 2 * f_w]
            H = ii[y + h, x + w]
            return (2 * G + 2 * B + D + E) - (2 * F + 2 * C + H + A)

        else:
            f_w = int(w / 2)
            f_h = int(h / 2)
            A = ii[y, x]
            B = ii[y, x + f_w]
            C = ii[y, x + w]
            D = ii[y + f_h, x]
            E = ii[y + f_h, x + f_w]
            F = ii[y + f_h, x + w]
            G = ii[y + h, x]
            H = ii[y + h, x + f_w]
            I = ii[y + h, x + w]
            return (2 * F + 2 * B + 2 * H + 2 * D) - (4 * E + C + G + A + I)
