import torch
import torch.nn as nn
import torch.optim as optim
from scr.data import get_dataloader
from scr.model import CNN, ModelConfig
from sklearn.metrics import accuracy_score

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_y_true = []
    all_y_pred = []

    for x,y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat,y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, predicted = torch.max(yhat, 1)
        all_y_true.extend(y.cpu().numpy())
        all_y_pred.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_y_true, all_y_pred)
    return avg_loss, accuracy

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = criterion(yhat, y)

            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(yhat, 1)
            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_y_true, all_y_pred)
    return avg_loss, accuracy


def train_model(config: ModelConfig, epochs=10, batch_size=16, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataloader = get_dataloader(batch_size=batch_size)

    model = CNN(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

if __name__ == "__main__":
    config = ModelConfig(
        filters=32,
        units1=128,
        units2=64,
        dropout_dense_rate=0.2,
        dropout_conv_rate=0.1,
        use_batchnorm=True,
        num_blocks=3,
        num_classes=2  
    )

    train_model(config, epochs=2, batch_size=16, lr=0.001)