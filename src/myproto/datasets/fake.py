from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader

def build_fake_dataset(name='fake', batch_size=32, num_workers=4, size=512, num_classes=10):
    transform = transforms.Compose([transforms.ToTensor()])
    ds = FakeData(size=size, image_size=(3,224,224), num_classes=num_classes, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
