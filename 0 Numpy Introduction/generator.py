import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.transform import rotate


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path: str, label_path: str, batch_size: int, image_size, rotation=False, mirroring=False, shuffle=False):

        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size

        
        self.file_path = file_path
        self.label_path = label_path

        self.batch_size = batch_size
        self.image_size = image_size
        
        # flags for different augmentations and whether the data should be shuffled for each epoch
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        with open(label_path) as f:
            image_dict = json.load(f)
        self.image_dict = image_dict

        # Note that the file names correspond to the dicts of the label dictionary.
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.image_data_number = int ( len(image_dict) )
        
        if self.shuffle == True:    # random shuffled order for all the pictures if desired
            self.image_sequence = np.array(list(image_dict.keys()))
            np.random.shuffle(self.image_sequence)
        else:
            self.image_sequence = np.arange(self.image_data_number)

        self.lastpicture_num = 0    # ongoing counter for all pictures

        




    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.


        # shape is: s, x, y, c = batch_size, resolution_x, resolution_y, channels
        images = np.zeros( (self.batch_size, self.image_size[0], self.image_size[1], 3) )
        labels = np.zeros(self.batch_size)


        for i in range(self.batch_size):

            # Note that your amount of total data might not be divisible without remainder with the batch_size:
            current_file_num = self.image_sequence[ self.lastpicture_num ]

            current_file = str( current_file_num )
            current_file_path = os.path.join(self.file_path, current_file + ".npy")
            image_data = np.load(current_file_path) # read the orginal image

            # resize to desired resolution:
            image_resized = resize(image_data, self.image_size, anti_aliasing=True)

            image_transformed = self.augment(image_resized) # call image transfomation function


            images[i, :, :, :] = image_transformed  # images
            labels[i] =  self.image_dict[ str(current_file_num) ] # labels as int

            self.lastpicture_num = (self.lastpicture_num + 1) % self.image_data_number # update initial seed for next file


        labels = labels.astype(int) # doesn't convert to int with int (...) for some reason

        return images, labels

    def augment(self,img):
        
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        # randomly flip if mirroring flag is set
        image_flipped = img
        if self.mirroring == True:
            if (np.random.randint(0,2) == 1):
                image_flipped = image_flipped[::-1, :] #flip up-down
            if (np.random.randint(0,2) == 1):
                image_flipped = image_flipped[:, ::-1] #flip left-right


        # rotate randomly if rotation flag is set:

        image_rotated = image_flipped
        if self.rotation == True:
            image_rotated = rotate( image_rotated, np.random.randint(0,4)*90 ) # 0 or 90 or 180 or 270 degree rotation

        img = image_rotated

        return img



    def class_name(self, x: int):
        # This function returns the class name for a specific input (int)

        return str (self.class_dict[x])


    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.

        images, labels = self.next()

        fig, axs = plt.subplots( int (np.ceil(self.batch_size / 3)) , 3)

        for i in range(self.batch_size):
            axs[int(np.floor(i/3)), i % 3].set_title(self.class_name(labels[i]))
            axs[int(np.floor(i/3)), i % 3].imshow(images[i,:,:,:])
            

        fig.tight_layout()

        plt.show()
        


