import cv2
import pkl_file_handler as pkl
from detector import ViolaJonesDetector
import random
from image_util import IntegralImage as ii
import imutils
import numpy as np


def build_sub_windows(image):
    sub_windows = []
    height, width = image.shape
    image_size = 19
    while image_size <= height and image_size <= width:
        x = 0
        while x + image_size <= width - 1:
            y = 0
            while y + image_size <= height - 1:
                sub_window = image[int(y):int(y + image_size), int(x):int(x + image_size)]
                sub_windows.append((sub_window, (int(x), int(y), int(image_size), int(image_size))))
                y += 1 * image_size
            x += 1 * image_size
        image_size *= 1.25
    return sub_windows

# Taken from StackOverflow for merging two sets together
def merge(lsts):
    sets = [set(lst) for lst in lsts if lst]
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets


if __name__ == "__main__":
    T, V = pkl.load("training.pkl"), pkl.load("test.pkl")
    total = T + V
    random.shuffle(total)
    total = ii.integrate_dataset(total)
    # face_detector = ViolaJonesDetector()
    # face_detector.train(total[0:1200], 0.65, 1.0, 0.001)
    # pkl.save("detector_new.pkl", face_detector)

    detector = pkl.load("detector.pkl")
    # detector.classifier_cascade[-1].strong_classifier.gamma += 2
    # detector.evaluate(total)

    # for image in total:
    #     if image[1] == 1:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        # frame = cv2.imread("hakim.jpg")
        image = frame#cv2.imread("hakim_no_glasses.jpg")#image[0]
        image = imutils.resize(image, width = 150)
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        window_frames = build_sub_windows(grey_image)
        window_frames = ii.integrate_dataset(window_frames)
        # window_frames = ii.variance_normalize_dataset(window_frames)
        # print(len(window_frames))
        inters_parti = []
        for frame, data in window_frames:
            # print(frame)
            if detector.classify(frame):
                # print("Found Face")
                num_inters = 0
                for partition in inters_parti:
                    for rectangle in partition:
                        if rectangle[0] <= data[0] + data[2] and rectangle[0] + rectangle[2] >= data[0] and rectangle[
                            1] <= \
                                data[1] + data[3] and rectangle[1] + rectangle[3] >= data[1]:
                            partition.add((data[0], data[1], data[2], data[3]))
                            num_inters += 1
                            break
                if num_inters == 0:
                    inters_parti.append({(data[0], data[1], data[2], data[3])})

        rectangles = merge(inters_parti)

        for partition in rectangles:
            partition = np.array(list(partition))
            x = int(np.mean(partition[:, 0]))
            y = int(np.mean(partition[:, 1]))
            w = int(np.mean(partition[:, 2]))
            h = int(np.mean(partition[:, 3]))
            image = cv2.rectangle(image, (x, y), (x + w, y + h),
                                  (255, 0, 0))
        # cv2.imshow("Detections", image)
        # Display the resulting frame
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
