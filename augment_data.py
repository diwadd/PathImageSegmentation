import random
import sys

import numpy as np

import data_handling as dh
import deep_models as dm

#random.seed(1)

if __name__ == "__main__":
    print("Starting main.")

    # ------------------------------------------------------------------------
    # Load data        
    # ------------------------------------------------------------------------

    train_images, truth_images = dh.read_data(data_dir="training/images/")
    iw, ih, ic = train_images[0].shape

    if len(train_images) != len(truth_images):
        sys.exit("ERROR: Dimension mismatch.")
    n_images = len(train_images)

    print("Data loaded.")
    print("Number of train images:" + str(len(train_images)))
    print("Number of truth images:" + str(len(truth_images)))
    print("Train image size: " + str(train_images[0].shape))
    print("Truth image size: " + str(truth_images[0].shape))

    # ------------------------------------------------------------------------
    # Augment data        
    # ------------------------------------------------------------------------

    ap=[[0.0, 1.0],
        [90.0, 1.0]] 
        #[180.0, 1.0],
        #[270., 1.0],
        #[0.0, 0.9],
        #[7.0, 0.9],
        #[3.5, 0.9],
        #[83.0, 0.9],
        #[86.5, 0.9],
        #[90.0, 0.9], 
        #[180.0, 0.9],
        #[270., 0.9]]

    dh.augment_data(train_images, 
                    truth_images,
                    ap=ap,
                    save_images=True)

