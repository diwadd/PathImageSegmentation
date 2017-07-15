import random
import data_handling as dh

#random.seed(1)

if __name__ == "__main__":
    print("Starting main.")

    # ------------------------------------------------------------------------
    # Load data        
    # ------------------------------------------------------------------------

    train_images, truth_images = dh.read_training_data()
    iw, ih, ic = train_images[0].shape

    print("Data loaded.")
    print("Number of train images:" + str(len(train_images)))
    print("Number of truth images:" + str(len(truth_images)))
    print("Train image size: " + str(train_images[0].shape))
    print("Truth image size: " + str(truth_images[0].shape))

    # ------------------------------------------------------------------------
    # Resize truth images        
    # ------------------------------------------------------------------------

    n_width = 100
    n_height = 100

    resized_truth_images = dh.resize_image_list(truth_images, n_width, n_height)
    print("Resized truth images.")
    print("Truth image after resize: " + str(truth_images[0].shape))

    # ------------------------------------------------------------------------
    # Convert resized truth images to
    # gray scale      
    # ------------------------------------------------------------------------

    grayscale_resized_truth_images = dh.convert_image_list_to_grayscale(resized_truth_images)
    print("Truth images converted to grayscale.")
    print("Truth image after conversion to grayscale: " + str(truth_images[0].shape))
    

    # ------------------------------------------------------------------------
    # Shuffle the data
    # ------------------------------------------------------------------------
    # We use zip to maintain the
    # correspondence image <-> truth.
    # truth_images is icluded so that we can
    # later compare back resized truth images
    # with the original truth images.
    # ------------------------------------------------------------------------

    train_truth_image_list = list(zip(train_images, grayscale_resized_truth_images, truth_images))
    random.shuffle(train_truth_image_list)
    train_images, grayscale_resized_truth_images, truth_images = zip(*train_truth_image_list)

    print("Data shuffled.")
    print("Number of train images:" + str(len(train_images)))
    print("Number of grayscale resized truth images:" + str(len(grayscale_resized_truth_images)))
    print("Train image size: " + str(train_images[0].shape))
    print("Grayscale resized truth image size: " + str(grayscale_resized_truth_images[0].shape))

    dh.plot_two_images(train_images[0], grayscale_resized_truth_images[0])

    # ------------------------------------------------------------------------
    # Back resize the truth data.
    # ------------------------------------------------------------------------

    back_resized_truth_images = dh.resize_image_list(grayscale_resized_truth_images, iw, ih)
    dh.plot_two_images(truth_images[0], back_resized_truth_images[0])

    # ------------------------------------------------------------------------
    # Scale the images to [0, 1]
    # ------------------------------------------------------------------------

    train_images = [train_images[i]/255.0 for i in range(len(train_images))]
    truth_images = [truth_images[i]/255.0 for i in range(len(truth_images))]





