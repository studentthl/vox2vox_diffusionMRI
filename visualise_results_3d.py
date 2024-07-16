import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import torch
from torch.utils.data import DataLoader
from model import GeneratorUNet, Discriminator
from dataset2 import get_train_val_datasets

def read_vox_file(file_path):
    with h5py.File(file_path, 'r') as file:
        data = file['data'][()]
    print(f"Shape of data read from {file_path}: {data.shape}")
    return data

def qspace_to_rspace(qspace_data):
    """
    Transform q-space data to r-space using the inverse Fourier transform.
    
    Args:
    qspace_data: A numpy array with the last dimension being the real and imaginary parts.
    
    Returns:
    rspace_data: The real part of the inverse Fourier-transformed data.
    """
    print(f"Shape of qspace_data before combining: {qspace_data.shape}")
    
    # Combine the real and imaginary parts into a complex array
    complex_data = qspace_data[..., 0] + 1j * qspace_data[..., 1]
    print(f"Shape of complex_data: {complex_data.shape}")

    # Apply the inverse Fourier transform
    rspace_data = np.fft.ifftn(complex_data)
    print(f"Shape of rspace_data after IFFT: {rspace_data.shape}")

    # Return the real part of the transformed data
    rspace_real_data = np.real(rspace_data)
    print(f"Shape of rspace_real_data (real part): {rspace_real_data.shape}")
    return rspace_real_data

def plot_isosurface(data, title):
    # Set up the mesh grid
    x, y, z = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), np.arange(data.shape[2]))

    # Calculate the threshold as the mean of the r-space data
    threshold = np.mean(data)
    print(f"Threshold value: {threshold}")

    # Extract the isosurface
    verts, faces, normals, values = measure.marching_cubes(data, level=threshold)

    # Plot the isosurface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)

    ax.set_title(title)
    plt.show()

# Load models
def load_models(generator_path, discriminator_path, device):
    generator = GeneratorUNet(in_channels=2, out_channels=2).to(device)
    discriminator = Discriminator(in_channels=2).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    generator.eval()
    discriminator.eval()
    return generator, discriminator

# Main function to execute the visualization
def main():
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator, discriminator = load_models("saved_models/qspace_3d/generator_3.pth",
                                           "saved_models/qspace_3d/discriminator_3.pth",
                                           device)
    
    # Load validation data
    file_paths = [f'Data/simulation_results_{i:02d}.mat' for i in range(1, 21)]
    _, val_dataset = get_train_val_datasets(file_paths)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Get a batch of validation data
    imgs = next(iter(val_loader))
    real_A = imgs["Undersampled_qspace"].to(device).permute(0, 4, 1, 2, 3)
    real_B = imgs["Q_space"].to(device).permute(0, 4, 1, 2, 3)
    
    # Generate images
    with torch.no_grad():
        fake_B = generator(real_A).cpu().numpy()
    
    # Convert to numpy
    real_A = real_A.cpu().numpy()
    real_B = real_B.cpu().numpy()
    
    # Transforming q-space to r-space
    rspace_real_A = qspace_to_rspace(real_A[0])
    rspace_real_B = qspace_to_rspace(real_B[0])
    rspace_fake_B = qspace_to_rspace(fake_B[0])
    
    # Plotting the isosurface
    plot_isosurface(rspace_real_A, 'Isosurface of R-Space Real A (Undersampled input)')
    plot_isosurface(rspace_real_B, 'Isosurface of R-Space Real B (Ground truth)')
    plot_isosurface(rspace_fake_B, 'Isosurface of R-Space Fake B (Generated output)')

if __name__ == "__main__":
    main()
