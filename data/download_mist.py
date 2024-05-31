import torchvision.datasets as datasets

# Download MNIST dataset in specified path
def download_mnist(data_path='./data'):
    print("Downloading MNIST dataset...")
    datasets.MNIST(root=data_path, train=True, download=True)
    datasets.MNIST(root=data_path, train=False, download=True)
    print("MNIST dataset downloaded.")


# ----------------------------
if __name__ == "__main__":
    download_mnist()