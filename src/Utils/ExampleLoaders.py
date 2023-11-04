from Imports import *

def split_im(array, nrows=128, ncols=128):
    """
    Split an image into sub-images. Takes a 3D input array (typically representing an image) and splits it
    into a grid of sub-matrices. The splitting is performed along the x and y axes (dimensions 1 and 2),
    creating nrows x ncols x channels for each sub-image.

    Args:
    - array (numpy.ndarray): The input 3D array to be split. This represents an image tensor.
    - nrows (int): The number of rows to split the image into.
    - ncols (int): The number of columns to split the image into.

    Returns:
    - numpy.ndarray: A 4D numpy array containing the sub-images. The first axis represents 
    the ith sub-image split from left to right and top to down.
    
    Note: This function assumes images of size or near size 512x512 for each channel.
    """
    array = array[:512,:512,:]
    r, h, ch = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols, ch)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols, ch))
    

def plot_subimages(array):
    """
    Plot sub-images as a grid of images.

    Args:
    - array (numpy.ndarray): 4D array where array.shape[0] = # of sub-images, 
    array.shape[1] = array.shape[2] = # values along x, y axes, and
    array.shape[2] = # channels.
        
    Returns:
    - None. 
    """
    num_ims = array.shape[0]
    # plot compartments
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, num_ims + 1):
        ax = fig.add_subplot(int(np.ceil(num_ims / int(np.sqrt(num_ims)))), int(np.sqrt(num_ims)), i)
        ax.imshow(array[i-1,:,:,0:3])
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=1)
        
def load_image(filename):
    ext = os.splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext == '.mat':
        im_dict = scipy.io.loadmat(filename)
        return Image.fromarray(im_dict['blurhyper'])
    else:
        return Image.open(filename)

def load_h5_dataset(filename="data/Dataset.h5"):
    h5 = h5py.File(filename,'r')

    GT_data = h5['groundtruth']
    mono_data = h5['mono']
    coord_data = h5['coordinate']
    info_data = h5['info']
    
    h5.close()

def load_blur_GT_pairs(dir):
    for filename in os.listdir(dir):
        if 'GT' in filename:
            return
        break