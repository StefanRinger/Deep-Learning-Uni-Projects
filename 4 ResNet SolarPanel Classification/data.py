from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import gray2rgb
from torchvision import transforms
import torch

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description

    def __init__(self, data, mode):

        self.data = data    # gather input as a dataframe (csv data)
        self.image = data.iloc[0:, 0]   #read first column of df into this. saves a lot of trouble compared to dealing with self.data only....
        self.label = data.iloc[0:, 1:]  #read second and third column into this

        self.mode = mode    # save the mode "train" or "val". This is needed since the returned sample image that is requested should be treated with different augmentation transformations, see below:

        self._transform = transforms.Compose([
            transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std)
        ])
        self._valdiation_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=train_mean, std=train_std)])



    def __len__(self):
        return len(self.image) # size produces super weird out of range bug in getitem ....

    def __getitem__(self, index):

        path = str(self.image.iloc[index])

        label = (self.label.iloc[index]).to_numpy().reshape(1,2)
        label_tensor = torch.as_tensor(label)

        image = imread(path)
        image_rgb = gray2rgb(image)

        if self.mode == 'train':
            image_augmented = self._transform(image_rgb)

        elif self.mode == 'val':
            image_augmented = self._valdiation_transform(image_rgb)


        return (image_augmented, label_tensor)