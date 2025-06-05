import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from google.colab import drive

drive.mount('/content/drive')
module_path = '/content/drive/MyDrive/Hung/Assignment-2/Source_code_MobileNetV2'
if module_path not in sys.path:
   sys.path.append(module_path)

from MobileNetV2_model import mobilenet_v2_scratch
from data_loader_4MBN import get_cifar_dataloaders
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use {DEVICE}")

CIFAR_DATA_PATH = "/content/drive/MyDrive/Hung/Assignment-2/cifar10_data/cifar-10-batches-py"

NUM_CLASSES = 10
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 25
print(f"\Prepare dataLoader with img_size={IMG_SIZE}, batch_size={BATCH_SIZE}")

train_loader, test_loader = get_cifar_dataloaders(
   cifar_batches_path=CIFAR_DATA_PATH,
   batch_size=BATCH_SIZE,
   img_size=IMG_SIZE,
   num_workers=2
)


print(f"\nInitializing MobileNetV2 with num_classes={NUM_CLASSES}, input_size={IMG_SIZE}...")
model = mobilenet_v2_scratch(num_classes=NUM_CLASSES, input_size=IMG_SIZE, width_mult=1.0)
model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
experiments_base_dir = "/content/drive/MyDrive/Hung/Assignment-2/Experiments"

print("\nBegin training process...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total_train
    epoch_acc = 100. * correct_train / total_train
    print(f"--- Complete epoch {epoch+1} ---")
    print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

    model.eval()
    correct_test = 0
    total_test = 0
    val_loss = 0.0

experiments_base_dir = "/content/drive/MyDrive/Hung/Assignment-2/Experiments/MobileNetV2_cifar_run"
model_filename = "mobilenetv2_try2.pth"

full_model_save_path = os.path.join(current_run_dir, model_filename)

try:
    torch.save(model.state_dict(), full_model_save_path)
    print(f"Model state_dict saved successfully at: {full_model_save_path}")
except Exception as e:
    print(f"Error saving model: {e}")
    import traceback
    traceback.print_exc()
