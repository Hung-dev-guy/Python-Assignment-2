import torch
import torch.nn as nn
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from MobileNetV2_model import mobilenet_v2_scratch 
from data_loader_4MBN import get_cifar_dataloaders, load_cifar10_data
from google.colab import drive
    
drive.mount('/content/drive') 
module_path = '/content/drive/MyDrive/Hung/Assignment-2/Source_code_MobileNetV2'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use: {DEVICE}")

SAVED_MODEL_PATH = "/content/drive/MyDrive/Hung/Assignment-2/Experiments/MobileNetV2_cifar_run/mobilenetv2_try1.pth"
CIFAR_DATA_PATH = "/content/drive/MyDrive/Hung/Assignment-2/cifar10_data/cifar-10-batches-py"

NUM_CLASSES = 10
IMG_SIZE = 224     
WIDTH_MULT = 1.0    
BATCH_SIZE = 16     

print(f"\nPreparing test dataLoader with img_size={IMG_SIZE}, batch_size={BATCH_SIZE}...")

_, test_loader = get_cifar_dataloaders(
    cifar_batches_path=CIFAR_DATA_PATH,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    num_workers=2, 
    shuffle_train=False 
)
print("Test DataLoader is ready")

_, _, _, _, label_names = load_all_cifar10_data(CIFAR_DATA_PATH) 
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f"\nInitializing MobileNetV2 architecture...")
model = mobilenet_v2_scratch(num_classes=NUM_CLASSES, input_size=IMG_SIZE, width_mult=WIDTH_MULT)
print("Initialized!")
model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE) 
model.eval()

# --- Bước 8: Vòng lặp Đánh giá ---
print("\nBegin evaluating process on test...")
total_loss = 0.0
correct_predictions = 0
total_samples = 0
all_labels = []
all_predictions = []

criterion = nn.CrossEntropyLoss() 

with torch.no_grad(): 
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

avg_test_loss = total_loss / total_samples
test_accuracy = 100. * correct_predictions / total_samples

print("\n--- Evaluation Result ---")
print(f"Average Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}% ({correct_predictions}/{total_samples})")

print("\nClassification Report:")
report = classification_report(all_labels, all_predictions, target_names=label_names[:NUM_CLASSES], zero_division=0)
print(report)

print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_predictions)
cm_df = pd.DataFrame(cm, index=label_names[:NUM_CLASSES], columns=label_names[:NUM_CLASSES])
print(cm_df)

import matplotlib.pyplot as plt
import seaborn as sns
try:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    cm_save_path = os.path.join(os.path.dirname(SAVED_MODEL_PATH), "confusion_matrix.png")
    plt.savefig(cm_save_path)
    print(f"Confusion matrix saved to {cm_save_path}")
    plt.show() 
except Exception as e:
    print(f"Cannot create confusion matrix: {e}")
