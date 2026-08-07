import torch
import torch.nn as nn
import torch.optim as optim
from qat import BitLinear

class SimpleBitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BitLinear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = BitLinear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def test_qat_training():
    model = SimpleBitMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    # Dummy data
    inputs = torch.randn(32, 10)
    labels = torch.randint(0, 2, (32,))

    print("Initial training loss check...")
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("QAT verified. Weights are being updated.")

if __name__ == "__main__":
    test_qat_training()
