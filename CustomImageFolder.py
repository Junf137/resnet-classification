from torchvision.datasets import ImageFolder
import re


# Custom dataset class that inherits from ImageFolder
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, is_valid_file=None):
        super(CustomImageFolder, self).__init__(root, transform, is_valid_file)

        self.max_fog_lv = self.calculate_max_fog_lv()

    def get_fog_lv(self, path):
        # Extract the fog level from file path
        idx_match = re.search(r"_(\d+)\.(png|PNG)$", path)
        return int(idx_match.group(1)) if idx_match else -1

    def calculate_max_fog_lv(self):
        max_fog_lv = -1
        for path, _ in self.samples:
            fog_lv = self.get_fog_lv(path)
            max_fog_lv = max(max_fog_lv, fog_lv)
        return max_fog_lv

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        fog_lv = self.get_fog_lv(path)
        return index, fog_lv, sample, target
