import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import random
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
import torch
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## CLASSIFIERS
class MLP(nn.Module):
  def __init__(self, input_size, num_classes=2):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, 100) # Fully connected layer with 100 hidden neurons
    self.fc2 = nn.Linear(100, num_classes) # Fully connected layer with num_classes outputs

  def forward(self, x):
    # x = x.view(-1, 768) # reshape the input tensor
    x = self.fc1(x)
    x = torch.relu(x)
    x = self.fc2(x)
    return x

class SimpleMLP(nn.Module):
  def __init__(self, input_size, num_classes=2):
    super(SimpleMLP, self).__init__()
    self.fc = nn.Linear(input_size, num_classes)

  def forward(self, x):
    x = self.fc(x)
    return x
  

def get_activations(layer, head, attentions_folder, split_ratio=0.8, seed=42):
    layers = ['layer9_attn', 'layer10_attn', 'layer11_attn', 'layer12_attn']

    # Set a random seed for reproducibility
    random.seed(seed)
    
    all_activations = []
    all_labels = []

    # List all .npy files in the attentions folder
    all_files = [f for f in os.listdir(attentions_folder) if f.endswith('.npy')]
    
    # Shuffle the files randomly
    random.shuffle(all_files)

    # Split the files into training and validation sets
    split_index = int(len(all_files) * split_ratio)
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    # Load training activations and labels
    for file in train_files:
        file_path = os.path.join(attentions_folder, file)
        activations = np.load(file_path)
        attn = activations[layers.index(layer), head, :]  # Extract attention for specified layer and head
        all_activations.append(attn)
        if 'female' in file:
            all_labels.append(1)
        elif 'male' in file:
            all_labels.append(0)

    # Load validation activations and labels
    for file in val_files:
        file_path = os.path.join(attentions_folder, file)
        activations = np.load(file_path)
        attn = activations[layers.index(layer), head, :]  # Extract attention for specified layer and head
        all_activations.append(attn)
        if 'female' in file:
            all_labels.append(1)
        elif 'male' in file:
            all_labels.append(0)

    # Convert the lists to NumPy arrays
    all_activations = np.array(all_activations)
    all_labels = np.array(all_labels)

    # Convert the NumPy arrays to PyTorch tensors
    train_activations_tensor = torch.tensor(all_activations[:split_index], dtype=torch.float32)
    val_activations_tensor = torch.tensor(all_activations[split_index:], dtype=torch.float32)

    train_labels = torch.tensor(all_labels[:split_index], dtype=torch.float32)
    val_labels = torch.tensor(all_labels[split_index:], dtype=torch.float32)

    return train_activations_tensor, val_activations_tensor, train_labels, val_labels


def get_activations_ethnicity(layer, head, attentions_folder, split_ratio=0.8, seed=42):
    layers = ['layer9_attn', 'layer10_attn', 'layer11_attn', 'layer12_attn']

    # Set a random seed for reproducibility
    random.seed(seed)
    
    all_activations = []
    all_labels = []

    # List all .npy files in the attentions folder
    all_files = [f for f in os.listdir(attentions_folder) if f.endswith('.npy')]
    
    # Shuffle the files randomly
    random.shuffle(all_files)

    # Split the files into training and validation sets
    split_index = int(len(all_files) * split_ratio)
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    ethnicity_mapping = {
    "EastAsian": 0,
    "Indian": 1,
    "Black": 2,
    "White": 3,
    "MiddleEastern": 4,
    "LatinoHispanic": 5,
    "SoutheastAsian": 6
    }
    keywords = sorted(ethnicity_mapping.keys(), key=len, reverse=True)

    # Load training activations and labels
    for file in train_files:
        file_path = os.path.join(attentions_folder, file)
        activations = np.load(file_path)
        attn = activations[layers.index(layer), head, :]  # Extract attention for specified layer and head
        all_activations.append(attn)

        for keyword in keywords:
            clean_keyword = keyword.strip("_")  # Remove underscore for matching
            if clean_keyword.lower() in file.lower():
                all_labels.append(ethnicity_mapping[keyword])
                break  # Stop checking once a match is found

    # Load validation activations and labels
    for file in val_files:
        file_path = os.path.join(attentions_folder, file)
        activations = np.load(file_path)
        attn = activations[layers.index(layer), head, :]  # Extract attention for specified layer and head
        all_activations.append(attn)
        
        for keyword in keywords:
            clean_keyword = keyword.strip("_")  # Remove underscore for matching
            if clean_keyword.lower() in file.lower():
                all_labels.append(ethnicity_mapping[keyword])
                break  # Stop checking once a match is found

    # Convert the lists to NumPy arrays
    all_activations = np.array(all_activations)
    all_labels = np.array(all_labels)

    # Convert the NumPy arrays to PyTorch tensors
    train_activations_tensor = torch.tensor(all_activations[:split_index], dtype=torch.float32)
    val_activations_tensor = torch.tensor(all_activations[split_index:], dtype=torch.float32)

    train_labels = torch.tensor(all_labels[:split_index], dtype=torch.float32)
    val_labels = torch.tensor(all_labels[split_index:], dtype=torch.float32)

    return train_activations_tensor, val_activations_tensor, train_labels, val_labels


def train_classifier(train_activations,train_labels,model):
  classifier = model
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Training loop
  num_epochs = 10
  for epoch in range(num_epochs):
    outputs = classifier(train_activations).to(device)
    train_labels = train_labels.long().to(device)
    loss = criterion(outputs, train_labels)

    preds = torch.argmax(outputs, dim=1)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

  print("Training complete!")
  return classifier


def evaluate_model(model, val_activations, val_labels, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = val_labels.size(0)  # Total number of samples

    all_preds = []  # Store predicted probabilities
    all_labels = []  # Store true labels

    with torch.no_grad():
        val_activations = val_activations.to(device)
        val_labels = val_labels.to(device)

        # Forward pass
        outputs = model(val_activations)
    
        loss = criterion(outputs, val_labels.long())  # Ensure labels are LongTensor
        
        total_loss += loss.item()
        
        # Get predicted classes
        _, preds = torch.max(outputs, 1)
        
        # Count correct predictions
        correct_predictions += (preds == val_labels).sum().item()

        # Calculate predicted probabilities using softmax
        probabilities = torch.softmax(outputs, dim=1)  # Shape: [batch_size, num_classes]

        # Store probabilities and labels for AP calculation
        all_preds.extend(probabilities[:, 1].cpu().numpy())
        all_labels.extend(val_labels.cpu().numpy())

    accuracy = correct_predictions / total_samples
    avg_loss = total_loss  # No averaging since there's only one forward pass

    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

    # Calculate Average Precision Score
    ap_score = average_precision_score(all_labels, all_preds, average='weighted')
    print(f'Average Precision Score: {ap_score:.4f}')
    print()

    return avg_loss, accuracy, ap_score, all_labels, all_preds


def train_all_heads(layer,attentions_folder,model):
    accuracies = []
    AP = []

    y_trues = []
    y_scores = []

    for head in range(12):
        train_activations, val_activations, train_labels, val_labels = get_activations(layer,head,attentions_folder)
        train_activations = train_activations.to(device)
        val_activations = val_activations.to(device)
        train_labels = train_labels.to(device)
        val_labels = val_labels.to(device)

        classifier = train_classifier(train_activations,train_labels,model)
        avg_loss, accuracy, ap_score, y_true, y_score = evaluate_model(classifier,val_activations, val_labels, nn.CrossEntropyLoss())
        
        accuracies.append(accuracy)
        AP.append(ap_score)

        y_trues.append(y_true)  # Convert to numpy for plotting
        y_scores.append(y_score)
    
    return accuracies, AP, y_trues, y_scores


def plot_auroc_curves(y_trues, y_scores, title):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))  # 3 rows, 4 columns for 12 heads
    axes = axes.ravel()  # Flatten the axes for easy indexing

    for i in range(12): 
        y_true = y_trues[i]
        y_score = y_scores[i]

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for the current head
        axes[i].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {roc_auc:.2f}')
        axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guess line
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('FPR')
        axes[i].set_ylabel('TPR')
        axes[i].set_title(f'Head {i+1}')
        axes[i].legend(loc="lower right")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_ap_curves(y_trues, y_scores, title):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))  # 3 rows, 4 columns for 12 heads
    axes = axes.ravel()  # Flatten the axes for easy indexing

    for i in range(12): 
        y_true = y_trues[i]
        y_score = y_scores[i]

        # Compute Precision-Recall curve and AP score
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap_score = average_precision_score(y_true, y_score)

        # Plot Precision-Recall curve for the current head
        axes[i].plot(recall, precision, color='blue', lw=2, label=f'AP = {ap_score:.2f}')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('Recall')
        axes[i].set_ylabel('Precision')
        axes[i].set_title(f'Head {i+1}')
        axes[i].legend(loc="lower left")

    # Adjust layout to prevent overlap
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def get_best_heads(head_classifier, layers, heads, ATTN_FOLDER, AP_THRESHOLD):
    all_dfs = []
    all_y_trues = []
    all_y_scores = []
    high_ap_dict = {} 

    for layer in layers:

        l = int(''.join(filter(str.isdigit, layer)))

        accuracies, AP, y_trues, y_scores = train_all_heads(layer,ATTN_FOLDER,head_classifier)
        
        all_y_trues.append(y_trues)
        all_y_scores.append(y_scores)

        df = pd.DataFrame({
        'Layer': layer,
        'Head': heads,
        'Accuracy': accuracies,
        'Average Precision': AP
        })
        all_dfs.append(df)

        # 🔍 Filter heads with AP > AP_THRESHOLD
        high_heads = df.loc[df["Average Precision"] > AP_THRESHOLD, "Head"].tolist()
        if high_heads:
            high_ap_dict[l] = high_heads

    return high_ap_dict, all_dfs