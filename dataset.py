import os
import re
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Precise control over undersampling using Gaussian masks in Fourier space
def gaussian_undersample_qspace(qspace, delta):
    rows, cols, slices, _ = qspace.shape  # Extract the shape considering the real and imaginary parts
    undersampled_qspace = np.zeros_like(qspace)
    
    for k in range(slices):
        mask = vardens_gaussian_sampling((rows, cols), delta, False)
        undersampled_slice_real = np.fft.fftshift(qspace[:, :, k, 0]) * mask
        undersampled_slice_imag = np.fft.fftshift(qspace[:, :, k, 1]) * mask
        undersampled_qspace[:, :, k, 0] = undersampled_slice_real
        undersampled_qspace[:, :, k, 1] = undersampled_slice_imag
    
    return undersampled_qspace

def vardens_gaussian_sampling(shape, delta, visualize_mask=False):
    import scipy.optimize
    import matplotlib.pyplot as plt
    
    c = 2 * np.sqrt(delta / np.pi)
    def equation(t):
        return scipy.special.erf(t) - c * t
    
    s = scipy.optimize.fsolve(equation, [1e-6, 1/c])[0]
    sigma = 1 / (s * np.sqrt(2))
    
    X, Y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))
    P = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    omega = np.random.rand(*P.shape) < P
    omega = np.fft.fftshift(omega)
    
    if visualize_mask:
        plt.figure()
        plt.imshow(omega, cmap='gray')
        plt.title('Gaussian Undersampling Mask')
        plt.axis('tight')
        plt.colorbar()
        plt.show()
    
    return omega


# Define weights initialization function
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class QRDataset(Dataset):
    def __init__(self, file_paths):
        self.voxel_data = []
        self.voxel_refs = []
        self.num_voxels = 0

        for file_path in file_paths:
            file = h5py.File(file_path, 'r')
            match = re.search(r'_(\d{2})\.mat$', file_path)
            suffix = match.group(1)
            dataset_name = f'results_{suffix}'
            results = file[dataset_name]

            self.num_voxels += results.shape[0]

            for idx in range(results.shape[0]):
                voxel_ref = results[idx, 0]
                self.voxel_refs.append((file_path, voxel_ref))
                voxel_data = file[voxel_ref]
                self.voxel_data.append(voxel_data)

    def __len__(self):
        return self.num_voxels

    def __getitem__(self, idx):
        file_path, voxel_ref = self.voxel_refs[idx]
        voxel_data = self.voxel_data[idx]

        rspace = voxel_data['rspace'][()]
        rspace_truth = torch.tensor(rspace.real, dtype=torch.float32)

        qspace = voxel_data['qspace'][()]
        qspace_combined = np.stack((qspace['real'], qspace['imag']), axis=-1)
        qspace_truth = torch.tensor(qspace_combined, dtype=torch.float32)

        # undersampled_image = self.gaussian_undersampling(qspace_combined)
        undersampled_image = self.gaussian_undersample_qspace(qspace_combined)

        return {"Undersampled_qspace": undersampled_image, "R_space": rspace_truth, "Q_space": qspace_truth}
    
    def gaussian_undersampling(self, qspace, sigma=5): #straightforward undersampling
        noise = np.random.randn(*qspace.shape) * sigma
        undersampled_qspace = qspace + noise
        return torch.tensor(undersampled_qspace, dtype=torch.float32)
    
    def gaussian_undersample_qspace(self, qspace, delta=5): #more precise
        return torch.tensor(gaussian_undersample_qspace(qspace, delta), dtype=torch.float32)


    def close(self):
        for voxel_data in self.voxel_data:
            voxel_data.file.close()


def get_train_val_datasets(file_paths, test_size=0.2):
    full_dataset = QRDataset(file_paths)
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=42)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    return train_dataset, val_dataset


if __name__ == "__main__":
    file_paths = [f'Data/simulation_results_{i:02d}.mat' for i in range(1, 21)]
    train_dataset, val_dataset = get_train_val_datasets(file_paths)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")

    # Example of loading a batch
    for batch in train_loader:
        print(batch['Undersampled_qspace'].shape, batch['R_space'].shape, batch['Q_space'].shape)
        break
    
