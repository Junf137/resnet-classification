from torchvision.datasets import ImageFolder
import re


# Custom dataset class that inherits from ImageFolder
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, is_valid_file=None):
        super(CustomImageFolder, self).__init__(root, transform, is_valid_file)

    def __getitem__(self, index):
        # Get the original tuple from ImageFolder
        path, target = self.samples[index]
        # Load the image
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # Extract the index from the file path
        idx_match = re.search(r"_(\d+)\.(png|PNG)$", path)
        idx = int(idx_match.group(1)) if idx_match else -1
        return idx, sample, target
