import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the neural network model
class NetworkAnomalyDetectionModel(nn.Module):
    def __init__(self, input_size=89):  # Ensure input_size is set to 89
        super(NetworkAnomalyDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)  # Output layer for binary classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add dropout to prevent overfitting
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return torch.sigmoid(self.fc4(x))


# Training func
def train_model(model, X_train, y_train, epochs=100, lr=0.0001):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_values = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)

        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    # Plot loss values and save as image
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss', color='blue')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save trained model
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved as 'trained_model.pth'")


if __name__ == "__main__":
    # Load preprocessed data directly
    X_train = torch.load('X_train.pt')
    y_train = torch.load('y_train.pt')


    # Define model
    model = NetworkAnomalyDetectionModel(input_size=X_train.shape[1])

    # Train the model
    train_model(model, X_train, y_train, epochs=100)
