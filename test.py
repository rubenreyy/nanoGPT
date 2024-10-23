import torch
import torch.nn as nn
from train import NetworkAnomalyDetectionModel

# Evaluation function
def evaluate_model(model, X_test, y_test, criterion, batch_size=64):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    num_batches = len(X_test) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        inputs, labels = X_test[start:end], y_test[start:end]
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted.squeeze() == labels).sum().item()
            total_samples += labels.size(0)

        if i % 10 == 0:
            print(f"Batch [{i}/{num_batches}], Loss: {loss.item():.4f}")
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / total_samples
    print(f"\nTest Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Load preprocessed data directly
    X_test = torch.load('X_test.pt')
    y_test = torch.load('y_test.pt')

    # Load the saved model
    model = NetworkAnomalyDetectionModel(input_size=X_test.shape[1])
    model.load_state_dict(torch.load('trained_model.pth'))

    # Define loss function for evaluation
    criterion = nn.BCELoss()

    # Evaluate the model on test data
    evaluate_model(model, X_test, y_test, criterion)
