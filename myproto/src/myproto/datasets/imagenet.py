from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def build_imagenet_dataset(root, batch_size=64, num_workers=8, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(root=f"{root}/train", transform=transform)
    val_ds = datasets.ImageFolder(root=f"{root}/val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
