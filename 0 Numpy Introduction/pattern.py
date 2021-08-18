import numpy as np
import matplotlib.pyplot as plt



# np.tile(), np.arange(), np.zeros(), np.ones(), np.concatenate() and np.expand dims()

class Checker:

    """ Implements a checkboard pattern.
    
    Inputs:

    - resolution (integer) = How big is the image/np array?
    - tile_size (integer) = How big are the tiles?

    Output:

    - np.array filled with 0 & 1 according to checkerboard pattern

     """


    def __init__(self, resolution: int, tile_size: int):

        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.array([])



    def draw(self):

        assert ( (self.resolution % (2 * self.tile_size)) == 0 ) # check if number of tiles * 2 can perfectly fill up resolution to avoid truncation of checkerboard


        number_of_repetitions = int( self.resolution / (2*self.tile_size) ) # how often is 2x2 base pattern repeated?
        #                                                                        0 1 . .    
        #                                                                        1 0 . .    
        #                                                                        . .   
        #                                                                        . .   
        
        zeroBlock = np.zeros( (self.tile_size, self.tile_size) )
        oneBlock = np.ones( (self.tile_size, self.tile_size) )

        base_pattern = np.hstack(       (  np.vstack((zeroBlock, oneBlock))    ,    np.vstack((oneBlock, zeroBlock))  )         )

        pattern = np.tile(base_pattern, (number_of_repetitions, number_of_repetitions) )

        self.output = pattern.copy()

        return pattern



    def show(self):

        plt.figure()
        plt.imshow(self.output, cmap="gray")
        plt.show()

        return







class Circle:


    """ Implements a circle pattern.
    
    Inputs:

    - resolution (integer) = How big is the image/np array?
    - radius (integer) = How big is the white circle?
    - position ( int tuple ) = Where is the center of the cirlce?

    Output:

    - np.array filled with 0 & 1 showing an according white cirle

     """


    def __init__(self, resolution: int, radius: int, position: tuple):


        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.array([])




    def draw(self):


        # idea: meshgrid & cartesian coordinates -> binary mask

        x = np.arange(0, self.resolution)
        y = np.arange(0, self.resolution)
        xx, yy = np.meshgrid(x, y, sparse=True)         # setup cartesian coordinate system
        distances = (xx - self.position[0]) **2 + (yy - self.position[1])**2     # calc distance from center point


        thresholded_distances = (distances < self.radius**2) # 1 if inside radius
        # threshold distance smaller than radius -> set to 1, rest 0 -> effectively a binary mask with the white circle


        self.output = thresholded_distances.copy()

        return thresholded_distances


    def show(self):

        plt.figure()
        plt.imshow(self.output, cmap="gray")
        plt.show()

        return


class Spectrum:


    """ Implements a Spectrum pattern.
    - Each channel intensity is from 0.0 to 1.0
    Inputs:
    - resolution (integer) = How big is the image/np array?

    Output:
    - RGB Spectrum

     """

    def __init__(self, resolution: int):
        self.resolution = resolution
        self.output = np.array([])

    def draw(self):
        myImage = np.ones([self.resolution,self.resolution,3])
        red = np.linspace(0.0,1.0,self.resolution)
        green= np.linspace(0.0,1.0,self.resolution)
        blue = np.flip(red)
        red = np.tile(red, (self.resolution, 1))
        green = np.tile(green, (self.resolution, 1)).T
        blue = np.tile(blue, (self.resolution, 1))
        myImage[:, :, 0] = red
        myImage[:, :, 1] = green
        myImage[:, :, 2] = blue
        self.output = myImage.copy()
        return myImage


    def show(self):

        plt.figure()
        plt.imshow(self.output, cmap="gray")
        plt.show()

        return





