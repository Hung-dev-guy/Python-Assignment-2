import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image

def load_cifar10_batch(file_path):
  with open(file_path, 'rb') as fo:
    batch_dict = pickle.load(fo, encoding='latin1')
  return batch_dict

def load_cifar10_data(data_folder_path):
  train_data = []
  train_labels = []
  for i in range(1,6):
    file_path = os.path.join(data_folder_path, f"data_batch_{i}")
    batch_dict = load_cifar10_batch(file_path)
    train_data.append(batch_dict['data'])
    train_labels.extend(batch_dict['labels'])

  X_train_raw = np.concatenate(train_data)
  X_train_reshaped = X_train_raw.reshape((len(X_train_raw), 3, 32, 32))
  X_train = X_train_reshaped.transpose(0, 2, 3, 1)
  y_train = np.array(train_labels)

  test_file_path = os.path.join(data_folder_path, 'test_batch')
  test_batch_dict = load_cifar10_batch(test_file_path)
  X_test_raw = test_batch_dict['data']
  X_test_reshaped = X_test_raw.reshape((len(X_test_raw), 3, 32, 32))
  X_test = X_test_reshaped.transpose(0, 2, 3, 1)
  y_test = np.array(test_batch_dict['labels'])

  label_names = []
  meta_file_path = "/content/drive/MyDrive/Hung/Assignment-2/cifar10_data/cifar-10-batches-py/batches.meta"
  with open(meta_file_path, 'rb') as fo:
      meta_dict = pickle.load(fo, encoding='latin1')
  label_names = meta_dict.get('label_names', [])


  return X_train, y_train, X_test, y_test, label_names

class Custom_Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Args </>
          data (numpy.ndarray): Image, expected in format (N, H, W, C).
          targets (numpy.ndarray): Labels for data (1D array).
          transform (callable, optional): A function/transform to apply to each sample.
        """

        if not (data.ndim == 4 and data.shape[1] == 32 and data.shape[2] == 32 and data.shape[3] == 3):
            raise ValueError(f"Image data is not in correct format (N, H, W, C). Actual shape: {data.shape}")

        self.data = data
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_np = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image_np)
        else:
            image = torch.tensor(image_np.transpose(2,0,1), dtype=torch.float32) / 255.0

        return image, label


def get_cifar_transforms(img_size=32, is_train=True):

    normalize_transform = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # CIFAR-10

    if is_train:
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize_transform
        ]
    else:
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize_transform
        ]
    return transforms.Compose(transform_list)

def get_cifar_dataloaders(cifar_batches_path, batch_size, img_size, num_workers=2, shuffle_train=True, validation_split_ratio=0.2, random_seed=42):

    X_train, y_train, X_test, y_test, lb_name = load_cifar10_data(cifar_batches_path)
    print(f"Loaded X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Loaded X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    X_train_new, X_val, y_train_new, y_val = None, None, None, None
    val_loader = None

    X_train_new, X_val, y_train_new, y_val = train_test_split(
            X_train,
            y_train,
            test_size=validation_split_ratio,
            random_state=random_seed,
            stratify=y_train
        )

    train_transforms = get_cifar_transforms(img_size=img_size, is_train=True)
    val_transforms = get_cifar_transforms(img_size=img_size, is_train=False)
    test_transforms = get_cifar_transforms(img_size=img_size, is_train=False)

    train_dataset = Custom_Dataset(data=X_train_new, targets=y_train_new, transform=train_transforms)
    test_dataset = Custom_Dataset(data=X_test, targets=y_test, transform=test_transforms)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    if X_val is not None and y_val is not None:
        val_dataset = Custom_Dataset(data=X_val, targets=y_val, transform=val_transforms)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Created Train DataLoader with {len(train_dataset)} samples, Test DataLoader with {len(test_dataset)} samples.")
    return train_loader, val_loader, test_loader, lb_name


if __name__ == "__main__":
  cifar_data_path = "/content/drive/MyDrive/Hung/Assignment-2/cifar10_data/cifar-10-batches-py"

  BATCH_SIZE = 16
  IMG_SIZE_FOR_MODEL = 224

  print(f"\nCreating dataloader with batch_size={BATCH_SIZE}, img_size={IMG_SIZE_FOR_MODEL}...")

  train_loader, val_loader, test_loader, _ = get_cifar_dataloaders(
      cifar_batches_path=cifar_data_path,
      batch_size=BATCH_SIZE,
      img_size=IMG_SIZE_FOR_MODEL
  )
  if val_loader is not None:
      images_val_batch, labels_val_batch = next(iter(val_loader))
      print(f"Shape of validation batch: {images_val_batch.shape}")
      print(f"Shape of validation batch: {labels_val_batch.shape}")
      print(f"Datatype of validation: {images_val_batch.dtype}")

# Made by Hung-dev-guy </>
