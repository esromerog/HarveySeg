import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def generate_cylinder():
    resolution = 10
    radius = 30
    height = 100

    # Generate linearly spaced values for theta and z
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, height, resolution)

    # Create a mesh grid for theta and z
    theta_grid, z_grid = np.meshgrid(theta, z)

    # Calculate x and y coordinates
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)

    # Flatten the arrays to create point cloud
    x_points = x_grid.ravel()
    y_points = y_grid.ravel()
    z_points = z_grid.ravel()

    # Stack the flattened arrays column-wise
    cylinder = np.column_stack((x_points, y_points, z_points))

    points = np.dot(cylinder, np.array([[0.58, 0.34, 0.58], [0.34, 0.8, 0.67], [0.30, 0.1, 0.4]]))

    return points

def get_tumor_center(tumor_mask):
    # Get coordinates of all tumor points
    tumor_points = np.column_stack(np.nonzero(tumor_mask))

    # Step 1: Calculate the center of mass of the tumor points
    center_of_mass = np.mean(tumor_points, axis=0)

    return center_of_mass

def get_tumor_mask(tumor_seg_file_path):
    # Load the tumor segmentation file
    tumor_seg_load = nib.load(tumor_seg_file_path)
    tumor_seg_voxels = tumor_seg_load.get_fdata()

    # Mask out the tumor in the image (considering tumor labels 1 and 4)
    tumor_mask = np.where((tumor_seg_voxels == 1) | (tumor_seg_voxels == 4), 1, 0)

    return tumor_mask

def get_route_score(tumor_mask, route_vector=None, plot = False):
    """
    Calculate the principal axes of the tumor and determine the score for the best alignment axis with a randomly generated route vector.
    
    Args:
        tumor_mask (array): Binary image with the tumor mask
    
    Returns:
        float: The score for the best alignment axis with the route vector.
    """
    
    # Get coordinates of all tumor points
    tumor_points = np.column_stack(np.nonzero(tumor_mask))

    # Step 1: Calculate the center of mass of the tumor points
    center_of_mass = np.mean(tumor_points, axis=0)
    print(center_of_mass)

    # Create an image space array to mark the center of mass
    center_of_mass_image = np.zeros(tumor_mask.shape)
    center_of_mass_indices = tuple(np.round(center_of_mass).astype(int))
    center_of_mass_image[center_of_mass_indices] = 1

    # Translate points to the center of mass
    translated_points = tumor_points - center_of_mass

    # Step 2: Calculate the inertia tensor for uniform density
    Ixx = np.sum(translated_points[:, 1]**2 + translated_points[:, 2]**2)
    Iyy = np.sum(translated_points[:, 0]**2 + translated_points[:, 2]**2)
    Izz = np.sum(translated_points[:, 0]**2 + translated_points[:, 1]**2)
    Ixy = Iyx = -np.sum(translated_points[:, 0] * translated_points[:, 1])
    Ixz = Izx = -np.sum(translated_points[:, 0] * translated_points[:, 2])
    Iyz = Izy = -np.sum(translated_points[:, 1] * translated_points[:, 2])

    inertia_tensor = np.array([
        [Ixx, Ixy, Ixz],
        [Iyx, Iyy, Iyz],
        [Izx, Izy, Izz]
    ])

    # Calculate eigenvalues and eigenvectors of the inertia tensor
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)

    # Principal axes are given by the eigenvectors
    principal_axes = eigenvectors

    # Step 3: Rotate points to align with principal axes
    rotated_points = np.dot(translated_points, principal_axes)

    # Step 4: Calculate the mass distribution along each principal axis
    mass_distribution = np.sum(np.abs(rotated_points), axis=0)

    # Calculate the route vector and its unit vector
    # route_vector = generate_random_route(tumor_mask, center_of_mass)

    if route_vector is None:
        route_hat_vector = generate_random_route(tumor_mask, center_of_mass)
    else:
        # Get the route vector from the parameters
        route_hat_vector = route_vector / np.linalg.norm(route_vector)

    # Normalize mass distribution
    mass_distribution_sum = np.sum(mass_distribution)
    rel_mass_distribution = mass_distribution / mass_distribution_sum

    # Compute projections of principal axes onto the route vector
    projections = np.abs(np.dot(principal_axes, route_hat_vector.T))

    print("Scores for each axis:", projections * rel_mass_distribution)
    
    # Determine the maximum score for alignment
    score = np.amax(projections * rel_mass_distribution)

    # Generate the plot if the user provides the option
    if plot: plot_main_axis(center_of_mass, tumor_points, principal_axes, rel_mass_distribution, projections)

    return score


def generate_random_route(tumor_mask, center_of_mass):
    # Create an enlarged route_mask with shape increased by 40 in each dimension
    enlarged_shape = tuple(dim + 40 for dim in tumor_mask.shape)
    route_mask = np.zeros(enlarged_shape)
    random_indices = tuple(np.random.randint(0, dim + 40) for dim in tumor_mask.shape)
    route_mask[random_indices] = 1

    # Determine the endpoint of the route vector
    end_point = np.column_stack(np.nonzero(route_mask))[0]

    # Calculate the route vector and its unit vector
    route_vector = end_point - center_of_mass

    return route_vector

def plot_main_axis(center_of_mass, tumor_points, principal_axes, route_vector, rel_mass_distribution, projections):
    # Step 7: Plot the points, principal axes, and ellipse
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F7EBDF')
    fig.patch.set_facecolor('#F7EBDF')

    # Plot original points
    ax.scatter(tumor_points[:, 0], tumor_points[:, 1], tumor_points[:, 2], c='k', marker='o', label='Tumor', s=0.05)

    # Plot the center of mass
    ax.scatter(center_of_mass[0], center_of_mass[1], center_of_mass[2], c='r', marker='x', s=100, label='Center of Mass')

    # Plot the route starting point
    ax.plot(route_vector[0], route_vector[1], route_vector[2], 'ro', label='Starting Point')

    # Plot the principal axes
    origin = center_of_mass
    for i in range(3):
        axis = principal_axes[:, i]
        scores = projections * rel_mass_distribution
        ax.quiver(origin[0], origin[1], origin[2], axis[0], axis[1], axis[2], length=15*rel_mass_distribution[i]*10, arrow_length_ratio=0.06, color=(1, 0, scores[i]))

    # Plot the ellipse
    # ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], ellipse_points[:, 2], color='m', label='Ellipse')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()

if __name__=='__main__':
    tumor_seg_file_path = "/Users/esromerog/Developer/Galen/Segmentation/FNL/GitRepo/Data/Raw/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz"
    tumor_mask = get_tumor_mask(tumor_seg_file_path)
    score = get_route_score(tumor_mask, plot=True)