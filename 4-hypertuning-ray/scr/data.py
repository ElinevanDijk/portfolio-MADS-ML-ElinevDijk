from pathlib import Path
import requests
import zipfile
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_hymenoptera():
    datadir = Path.home() / ".cache/mads_datasets/hymenoptera_data"
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    if not datadir.exists():
        logger.info(f"Creating directory {datadir}")
        datadir.mkdir(parents=True)

        response = requests.get(url)
        zip_file_path = datadir / "hymenoptera_data.zip"
        with open(zip_file_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Extracting {zip_file_path}")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(datadir)
        zip_file_path.unlink()
    else:
        logger.info(f"Directory {datadir} already exists, skipping download.")
    return datadir / "hymenoptera_data"


def get_dataloader(batch_size=16, val_split = 0.2):
    datadir = get_hymenoptera()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], #based on averages pixels ImageNet
                             [0.229, 0.224, 0.225]) #based on std pixels ImageNet
    ])
    dataset = datasets.ImageFolder(datadir / "train", transform=transform)
    val_size = int(len(dataset)* val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
