import cv2
import numpy as np

import matplotlib.pyplot as plt
import data_handling as dh

if __name__ == "__main__":

    file_name = "augmented_images/numpy_image_array_id_0_0p0_30_30_0p85.npz"

    loaded_data = np.load(file_name)

    image = (255.0*loaded_data["image"]).astype(np.uint8)
    label = (255.0*loaded_data["label"]).astype(np.uint8)
    label = np.reshape(label, (50, 50))

    print("label shape: " + str(label.shape))

    label = cv2.resize(label,
                       (500, 500),
                       interpolation = cv2.INTER_LINEAR)
    mask = cv2.imread("augmented_images/truth_image_id_0_0p0_30_30_0p85.png")

    dh.plot_two_images(image, label, label)



