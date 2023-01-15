import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from moe2 import MoE
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 50

#
num_experts = 200
hidden_size = 200
k = 8

# Load MNIST dataset
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

#models
hydra = MoE(input_size, num_classes, num_experts, hidden_size, k=k, noisy_gating=True, mlp = False).to(device)
bench = MoE(input_size, num_classes, num_experts, hidden_size, k=k, noisy_gating=True, mlp = True).to(device)

# Train the model
total_step = len(train_loader)
eval_dict = {}
for i,model in enumerate([hydra]):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    evals = []
    for epoch in range(num_epochs):
        for z, (images, labels) in enumerate(train_loader):
            # Reshape images to (batch_size, input_size)
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs[0], labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)

        with torch.no_grad():
            correct = 0
            total = 0
            cc = 0
            for images, labels in test_loader:
                images = images.reshape(-1, input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)[0] 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            evals.append(correct/total)


            print(f'Accuracy of the model on the 10000 test images: {correct/total} %')

    eval_dict[i] = evals


from matplotlib import pyplot as plt
for i in [0,1]:
    plt.scatter(range(len(eval_dict[i])), eval_dict[i])

import pandas as pd 
pd.Series(eval_dict[0]).mean()
pd.Series(eval_dict[1]).mean()#hydra
