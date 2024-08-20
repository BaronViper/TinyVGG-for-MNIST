import torch
import torchvision
import random
import mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from tqdm.auto import  tqdm


print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = torchvision.datasets.MNIST(root="data",
                                           train=True,
                                           download=True,
                                           transform=ToTensor(),
                                           target_transform=None
                                           )

test_dataset = torchvision.datasets.MNIST(root="data",
                                          train=False,
                                          download=True,
                                          transform=ToTensor(),
                                          target_transform=None
                                          )

class_names = train_dataset.classes
class_to_idx = train_dataset.class_to_idx

# Show random images from MNIST
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
  sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
  img, label = train_dataset[sample_idx]
  figure.add_subplot(rows, cols, i)
  plt.title(class_names[label])
  plt.imshow(img.squeeze(), cmap='gray')
  plt.axis("off")


BATCH_SIZE = 32
train_dataloader = DataLoader(dataset= train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(dataset= test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CNN Model based on TinyVGG
class MNISTModelV0(torch.nn.Module):
  def __init__(self,
               input_shape,
               hidden_units,
               output_shape):
    super().__init__()

    self.conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=hidden_units,
                   kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                   kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
    )

  def forward(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.classifier(x)
    return x

model_1 = MNISTModelV0(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

# Loss, Optim, Accuracy Functions
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)

accuracy = Accuracy(task="multiclass", num_classes=len(class_names))

# TRAINING STEP
EPOCHS = 5

for epoch in tqdm(range(EPOCHS)):
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        model_1.train()

        y_pred = model_1(X)
        loss = loss_fn(y_pred, y)

        train_loss += loss
        train_acc += accuracy(y_pred.argmax(dim=1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataset)} samples")
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}%")


def make_predictions(model, data, device=device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)

# Get test samples and labels for display
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_dataset), k=9):
  test_samples.append(sample)
  test_labels.append(label)

# Get predictions of samples
pred_probs = make_predictions(model=model_1, data=test_samples)
pred_classes = pred_probs.argmax(dim=1)


# Plot sample results
plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i, sample in enumerate(test_samples):
  plt.subplot(rows, cols, i + 1)

  plt.imshow(sample.squeeze(), cmap="gray")

  pred_label = class_names[pred_classes[i]]
  y_label = class_names[test_labels[i]]

  title_text = f"Pred: {pred_label} | Truth: {y_label}"

  if pred_label == y_label:
    plt.title(title_text, fontsize=10, c="g")
  else:
    plt.title(title_text, fontsize=10, c="r")
  plt.axis("off")


# Confusion Matrix over whole test dataset

y_preds = []
model_1.eval()

with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making Predictions..."):
    X, y = X.to(device), y.to(device)

    y_logit = model_1(X)

    y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)

    y_preds.append(y_pred.cpu())

y_pred_tensor = torch.cat(y_preds)


confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor, target=test_dataset.targets)


fig, ax = plot_confusion_matrix(
    conf_mat = confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)