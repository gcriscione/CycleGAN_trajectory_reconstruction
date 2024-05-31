import matplotlib.pyplot as plt
import os

SAVE_DIR = "result/plots"

# Plot original, noisy, and reconstructed images. Optionally save the plots to a directory.
def plot_images(original, noisy, reconstructed, num_images=5, show=True, save=False, epoch=None, extends_name=None, model_seed=None):
    num_images = min(num_images, original.shape[0])
    
    # Create the directory if it does not exist
    if save and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        original_image = original[i].numpy().squeeze()
        noisy_image = noisy[i].numpy().squeeze()
        reconstructed_image = reconstructed[i].numpy().squeeze()

        plt.subplot(3, num_images, i + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(noisy_image, cmap='gray')
        plt.title("Noisy")
        plt.axis('off')
        
        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    
    # Save the plot if save flag is True
    if save:
        file_name = "plot"
        if model_seed is not None:
            file_name += f"_seed{model_seed}"
        if epoch is not None:
            file_name += f"_epoch{epoch}"
        if extends_name is not None:
            file_name += f"_{extends_name}"
        file_name += ".png"
        plt.savefig(os.path.join(SAVE_DIR, file_name))
    
    # Show the plot if show flag is True
    if show:
        plt.show()
    else:
        plt.close()