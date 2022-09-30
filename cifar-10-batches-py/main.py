from resnet.resnet import ResNet, ResidualBlock
import torch
import numpy as np
import torch.nn as nn
import cv2
import os

model = ResNet(ResidualBlock, [2, 2, 2])
learning_rate = 0.01
num_epochs = 10
device = 'cpu'

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

images = []
all_images = []
for dir in ["training batch 1 of 5", "training batch 2 of 5", "training batch 3 of 5",
            "training batch 4 of 5", "training batch 5 of 5"]:
    for file in os.listdir(dir):
        if file.endswith(".png"):
            im = cv2.imread(os.path.join(dir, file))
            images.append(im.reshape(3, 32, 32))
            if len(images) == 100:
                all_images.append(images)
                images = []
        elif file.endswith(".txt"):
            with open(os.path.join(dir, file)) as f:
                labels_str = f.readlines()
            labels = [int(lab) for lab in labels_str]

labels_100 = []
all_labels = []
for label in labels:
    labels_100.append(label)
    if len(labels_100)==100:
        all_labels.append(labels_100)
        labels_100 = []

all_labels = np.array(all_labels).astype('int')
all_images = np.array(all_images)

total_step = 100

curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(zip(all_images, all_labels)):
        images = torch.from_numpy(images).to(device)
        labels = torch.as_tensor(labels).to(device)

        # Forward pass
        outputs = model(images.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

images = []
testing_images = []

for dir in ["testing batch 1 of 1"]:
    for file in os.listdir(dir):
        if file.endswith(".png"):
            im = cv2.imread(os.path.join(dir, file))
            images.append(im.reshape(3, 32, 32))
            if len(images) == 100:
                testing_images.append(images)
                images = []
        elif file.endswith(".txt"):
            with open(os.path.join(dir, file)) as f:
                labels_str = f.readlines()
            labels = [int(lab) for lab in labels_str]

labels_100 = []
testing_labels = []
for label in labels:
    labels_100.append(label)
    if len(labels_100)==100:
        testing_labels.append(labels_100)
        labels_100 = []

testing_labels = np.array(testing_labels).astype('int')
testing_images = np.array(testing_images)


# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in zip(testing_images, testing_labels):
        images = torch.from_numpy(images).to(device)
        labels = torch.as_tensor(labels).to(device)
        outputs = model(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')
