import unittest
from detrain.ppl.dataset_util import get_torchvision_dataset

class TestDataset(unittest.TestCase):

    def test_imagenet_loader(self):
        (train_dataloader, test_dataloader) = get_torchvision_dataset("ImageNet", 64)
        print(train_dataloader, test_dataloader)
    def test_mnist_loader(self):
        (train_dataloader, test_dataloader) = get_torchvision_dataset("MNIST", 64)
        print(train_dataloader, test_dataloader)

if __name__=="__main__":

   unittest.main()