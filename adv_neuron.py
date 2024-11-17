import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Define the SimpleNN sub-network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

    def forward(self, x):
        return self.model(x)

class DeepWeightsLayer(nn.Module):
    def __init__(self, input, units):
        super(DeepWeightsLayer, self).__init__()
        self.weights = [SimpleNN() for i in range(input)]
        self.bias = nn.Parameter(torch.zeros(units))

    def forward(self, x):
        print('test x.T:',x.T.shape)
        mat = [wi(b_xi) for wi, b_xi in zip(self.weights, x.T)]
        return mat.T+self.bias



# Step 2: Define the Baseline Model
class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm([28*28]),
            nn.Linear(28 * 28, 90),
            nn.ReLU(),
            nn.LayerNorm([90]),
            nn.Linear(90, 66),
            nn.ReLU(),
            nn.LayerNorm([66]),
            nn.Linear(66, 10),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.model(x)
class DeepLinear(nn.Module):
    def __init__(self, input, units):
        super(DeepLinear, self).__init__()
        self.w = {
            'w1': nn.Parameter(torch.empty(1, input, units, 2).uniform_(-2.0, 2.0)),
            'b1': nn.Parameter(torch.empty(1, input, units, 2).uniform_(-2.0, 2.0)),
            'w21': nn.Parameter(torch.empty(1, input, units, 2).uniform_(-2.0, 2.0)),
            'w22': nn.Parameter(torch.empty(1, input, units, 2).uniform_(-2.0, 2.0)),
            'b21': nn.Parameter(torch.empty(1, input, units, 1).uniform_(-2.0, 2.0)),
            'b22': nn.Parameter(torch.empty(1, input, units, 1).uniform_(-2.0, 2.0)),
            'w3': nn.Parameter(torch.empty(1, input, units, 2).uniform_(-2.0, 2.0)),
            'b3': nn.Parameter(torch.empty(1, input, units, 1).uniform_(-2.0, 2.0)),
            }
        self.bias = nn.Parameter(torch.zeros(units))


        # Add normalization layers
        self.norm0 = nn.LayerNorm([input])
        self.norm1 = nn.LayerNorm([input, units, 2])
        self.norm2 = nn.LayerNorm([input, units, 2])
        self.norm3 = nn.LayerNorm([input, units, 1])
        
        self.n_params = sum([i.numel() for i in self.w.values()])+self.bias.numel()
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.w.items():
            if 'w' in name:  # For weights
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'b' in name:  # For biases
                nn.init.constant_(param, 0.0)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        x = self.norm0(x).unsqueeze(-1).unsqueeze(-1) # Shape: (32, 784, 1, 1)
        l1 = nn.LeakyReLU()( self.norm1(x*self.w['w1']+self.w['b1']) )
        l21 = (l1*self.w['w21']).sum(dim=-1, keepdim=True) + self.w['b21']
        l22 = (l1*self.w['w22']).sum(dim=-1, keepdim=True) + self.w['b22']
        l2 = nn.LeakyReLU()( self.norm2( torch.cat((l21, l22), dim=-1)))
        l3 = (l2*self.w['w3']).sum(dim=-1, keepdim=True) + self.w['b3']
        out = self.norm3(l3)+x
        out = nn.LeakyReLU()(out.sum(-3) + self.bias.unsqueeze(-1)).squeeze()
        return out

# Step 3: Define the SubNetwork-Based Model
class SubNetworkModel(nn.Module):
    def __init__(self):
        super(SubNetworkModel, self).__init__()
        self.flatten = nn.Flatten()
        #self.input_to_hidden = [SimpleNN() for i in range(128)]
        #self.hidden_to_output = [SimpleNN()]
        self.l1 = nn.Linear(784, 64)
        #self.l1 = DeepLinear(784, 64)

        #self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 10)
        self.l2 = DeepLinear(64, 32)
        #self.l3 = DeepLinear(32, 10)
        #self.output_layer = nn.Linear(128, 10)  # Final layer

    def forward(self, x):
        x = self.flatten(x)
        l1 = nn.GELU()(self.l1(x))
        l2 = nn.GELU()(self.l2(l1))
        l3 = self.l3(l2)
        return nn.Softmax()(l3)

# Step 4: Training and Evaluation
def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.0003):
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, accuracy: {accuracy * 100:.2f}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Step 5: Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Step 6: Run Baseline and SubNetwork Models
'''
print("\nTraining SubNetwork-Based Model...")
subnetwork_model = SubNetworkModel()
subnetwork_accuracy = train_and_evaluate(subnetwork_model, train_loader, test_loader)
'''
print("Training Baseline Model...")
baseline_model = BaselineModel()
baseline_accuracy = train_and_evaluate(baseline_model, train_loader, test_loader)

# Compare Results
print(f"Baseline Model Accuracy: {baseline_accuracy * 100:.2f}%")
print(f"SubNetwork Model Accuracy: {subnetwork_accuracy * 100:.2f}%")
