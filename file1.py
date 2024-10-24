import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Prompt for the number of institutions
num_institutions = int(input("Enter the number of institutions: "))

institutions = {}

# Step 2: Take input for each institution's name and dataset
for i in range(num_institutions):
    name = input(f"Enter the name of institution {i + 1}: ")
    dataset_path = input(f"Enter the dataset path for {name} (CSV file): ")
    try:
        data = pd.read_csv(dataset_path)
        institutions[name] = data
    except FileNotFoundError:
        print(f"Error: File at {dataset_path} not found. Please check the path.")
        exit()

# Step 3: Define a simple Neural Network for each institution
class InstitutionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(InstitutionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 4: Preprocess and create local datasets for each institution
def preprocess_data(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return torch.Tensor(X), torch.Tensor(y)

institution_datasets = []
institution_names = []

# Preprocess first dataset to get X's shape and use it to initialize the global model later
first_data_processed = False

for name, data in institutions.items():
    X, y = preprocess_data(data)
    if not first_data_processed:
        input_size = X.shape[1]
        first_data_processed = True
    dataset = TensorDataset(X, y)
    institution_datasets.append(dataset)
    institution_names.append(name)

# Step 5: Initialize Global Model and Send Manipulated Initial Weights
attribute_names = [
    "No. of DSA questions", "CGPA", "Knows ML", "Knows DSA", 
    "Knows Python", "Knows JavaScript", "Knows HTML", "Knows CSS",
    "Was in Coding Club", "No. of backlogs", "Placement Package", 
    "Choose your gender", "Your current year of Study", 
    "Do you have Depression?", "Do you have Anxiety?", 
    "Do you have Panic attack?", "Did you seek any specialist for treatment?"
]

# Define the weight importance per attribute (manually adjusted)
attribute_weights = {
    "Placement Package": 1.0,    # Highest weight
    "CGPA": 1.0,                 # Highest weight
    "No. of DSA questions": 0.8, # Lesser weight
    "Knows ML": 0.6, "Knows DSA": 0.6, "Knows Python": 0.6,
    "Knows JavaScript": 0.6, "Knows HTML": 0.6, "Knows CSS": 0.6, # Even lesser weight
    "Was in Coding Club": 0.4,   # Lesser importance
    "No. of backlogs": -0.5,     # Inversely weighted
    "Choose your gender": 0.2,   # Lesser importance
    "Your current year of Study": 0.2,
    "Do you have Depression?": 0.2,
    "Do you have Anxiety?": 0.2,
    "Do you have Panic attack?": 0.2,
    "Did you seek any specialist for treatment?": 0.2
}

# Adjust the initial weights based on attribute importance
def manipulate_initial_weights(model, attribute_weights):
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            # Adjust only the first layer's weights based on attribute importance
            if idx == 0:
                # Apply weight adjustments based on the order of attributes
                for i in range(param.size(1)):
                    attribute_name = attribute_names[i] if i < len(attribute_names) else "Unknown"
                    importance = attribute_weights.get(attribute_name, 0.1)  # Default lesser weight
                    param[:, i] *= importance  # Adjust weights per attribute
    return model.state_dict()

# Initialize the global model and manipulate initial weights
global_model = InstitutionModel(input_size=input_size, output_size=1)
initial_weights = manipulate_initial_weights(global_model, attribute_weights)

# Step 6: Local Training and FedAvg (rest of the code remains unchanged)
def train_local_model(local_model, data_loader, epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(local_model.parameters(), lr=0.001)

    local_model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

def fed_avg(global_weights, local_weights_list):
    new_weights = {}
    for key in global_weights.keys():
        new_weights[key] = torch.mean(torch.stack([local_weights[key] for local_weights in local_weights_list]), dim=0)
    return new_weights

local_models = [InstitutionModel(input_size=input_size, output_size=1) for _ in institution_datasets]
for local_model in local_models:
    local_model.load_state_dict(initial_weights)

data_loaders = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in institution_datasets]
local_weights_list = []

# Train each local model
for local_model, data_loader in zip(local_models, data_loaders):
    train_local_model(local_model, data_loader)
    local_weights_list.append(local_model.state_dict())

# Step 7: Perform FedAvg
global_weights = fed_avg(global_model.state_dict(), local_weights_list)
global_model.load_state_dict(global_weights)

# Step 8: Define convergence check and stopping criterion
def has_converged(global_weights, new_global_weights, threshold=1e-3):
    total_diff = 0
    for key in global_weights.keys():
        total_diff += torch.sum(torch.abs(global_weights[key] - new_global_weights[key])).item()
    return total_diff < threshold

converged = False
max_rounds = 100
round_num = 0
threshold = 1e-3

while not converged and round_num < max_rounds:
    local_weights_list = []
    for local_model, data_loader in zip(local_models, data_loaders):
        train_local_model(local_model, data_loader)
        local_weights_list.append(local_model.state_dict())

    new_global_weights = fed_avg(global_model.state_dict(), local_weights_list)
    converged = has_converged(global_model.state_dict(), new_global_weights, threshold=threshold)
    global_model.load_state_dict(new_global_weights)
    round_num += 1
    print(f"Round {round_num} completed. Converged: {converged}")

# Step 9: Use Borda Count for Ranking Institutions
def borda_count(local_weights_list):
    scores = [sum([torch.sum(weights[key]).item() for key in weights]) for weights in local_weights_list]
    sorted_indices = np.argsort(scores)[::-1]
    return sorted_indices

institution_ranks = borda_count(local_weights_list)
ranked_institutions = [(institution_names[i], rank + 1) for rank, i in enumerate(institution_ranks)]
print("Institution Rankings: ", ranked_institutions)

# Example Output
for institution, rank in ranked_institutions:
    print(f"{institution} is ranked {rank}")
