from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

#  Utility functions to return training and testing dataset.
def get_torchvision_dataset(name, batch_size, distributed=False):
    if (name == "MNIST" and distributed == False):
        # This dataset is used for examples

        # Dataset for training
        training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
        )

        # Dataset for evaluation step
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        return (train_dataloader, test_dataloader)
    elif (name == "ImageNet" and distributed == False):
        #Prepare transformations for data augmentation
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Dataset for training
        train_dataset = datasets.ImageNet(
            root='data', 
            split="train",
            transform=transform
        )

        # Dataset for evaluation step
        test_dataset = datasets.ImageNet(
            root='data', 
            split="eval",
            transform=transform
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        return (train_dataloader, test_dataloader)
    elif (name == "MNIST" and distributed == True):
        # Support DP: the training dataset will be sharded across multiple devices.
        # Not implemented yet.
        training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        return (train_dataloader, test_dataloader)
         

