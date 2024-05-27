import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_vox_file(file_path):
    with h5py.File(file_path, 'r') as file:
        data = file['data'][()]

    return data

def qspace_to_rspace(qspace_data):
    """
    Transform q-space data to r-space using the inverse Fourier transform.
    
    Args:
    qspace_data: A numpy array with the last dimension being the real and imaginary parts.
    
    Returns:
    rspace_data: The real part of the inverse Fourier-transformed data.
    """
    # Remove the first dimension (batch size) if present
    if qspace_data.shape[0] == 1:
        qspace_data = qspace_data[0]

    # Combine the real and imaginary parts into a complex array
    complex_data = qspace_data[0] + 1j * qspace_data[1]

    # Apply the inverse Fourier transform
    rspace_data = np.fft.ifftn(complex_data)
 

    # Return the real part of the transformed data
    rspace_real_data = np.real(rspace_data)
    return rspace_real_data

def plot_voxel_data(data, title, threshold=0.5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Create a binary mask
    mask = data > threshold

    # Get the coordinates of the voxels
    x, y, z = np.where(mask)

    # Plot the voxels
    ax.scatter(x, y, z, zdir='z', c='red')

    ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    # Example usage:
    real_A = read_vox_file('images/qspace_3d/epoch_1_real_undersampled_A.vox')
    real_B = read_vox_file('images/qspace_3d/epoch_1_real_qspace_B.vox')
    fake_B = read_vox_file('images/qspace_3d/epoch_1_fake_qspace_B.vox')

    # Transforming q-space to r-space
    rspace_real_A = qspace_to_rspace(real_A)
    rspace_real_B = qspace_to_rspace(real_B)
    rspace_fake_B = qspace_to_rspace(fake_B)

    # Plotting the data
    plot_voxel_data(rspace_real_A, 'R-Space Real A. The undersampled input to the generator.')
    plot_voxel_data(rspace_real_B, 'R-Space Real B. The target or ground truth image. ')
    plot_voxel_data(rspace_fake_B, 'R-Space Fake B. The output of the generator.')



    '''epoch_0_fake_B.vox: The output of the generator network when given the undersampled input. This is what the generator is trying to make look like the real B image.

    epoch_0_real_A.vox: The undersampled input to the generator. This is the "noisy" version of the image that the generator is supposed to "denoise" or reconstruct to look like real B.

    epoch_0_real_B.vox: The target or ground truth image. This is what the generator's output (fake B) should look like ideally.'''