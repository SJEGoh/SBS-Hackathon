import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GNN_model import CrowdGNN

T = 20 # number of timestamps
N = 5 # number of nodes
F = 4  # crowding, sin_time, cos_time, discount, number of features

"""
ToDo:
- Extras: Factors we know at every current timestep (eg. time, discount)
"""
# 
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4],
    [1, 0, 2, 1, 3, 2, 4, 3]
], dtype = torch.long)

# Time steps
X = np.array([
# t0
[[0.30, 0.00, 1.00, 0],
 [0.35, 0.00, 1.00, 0],
 [0.40, 0.00, 1.00, 0],
 [0.32, 0.00, 1.00, 0],
 [0.28, 0.00, 1.00, 0]],

# t1
[[0.40, 0.31, 0.95, 0],
 [0.45, 0.31, 0.95, 0],
 [0.55, 0.31, 0.95, 0],
 [0.48, 0.31, 0.95, 0],
 [0.42, 0.31, 0.95, 0]],

# t2
[[0.55, 0.59, 0.81, 0],
 [0.65, 0.59, 0.81, 0],
 [0.80, 0.59, 0.81, 0],
 [0.70, 0.59, 0.81, 0],
 [0.60, 0.59, 0.81, 0]],

# t3 (peak)
[[0.70, 0.81, 0.59, 0],
 [0.85, 0.81, 0.59, 0],
 [0.95, 0.81, 0.59, 0],
 [0.88, 0.81, 0.59, 0],
 [0.75, 0.81, 0.59, 0]],

# t4 (discount starts)
[[0.65, 0.95, 0.31, 1],
 [0.78, 0.95, 0.31, 1],
 [0.85, 0.95, 0.31, 1],
 [0.80, 0.95, 0.31, 1],
 [0.68, 0.95, 0.31, 1]],

# t5
[[0.55, 1.00, 0.00, 1],
 [0.65, 1.00, 0.00, 1],
 [0.70, 1.00, 0.00, 1],
 [0.68, 1.00, 0.00, 1],
 [0.58, 1.00, 0.00, 1]],

# t6
[[0.48, 0.95, -0.31, 1],
 [0.55, 0.95, -0.31, 1],
 [0.60, 0.95, -0.31, 1],
 [0.58, 0.95, -0.31, 1],
 [0.50, 0.95, -0.31, 1]],

# t7–t19 (repeat off-peak → rising again)
])


X = torch.tensor(np.concatenate([X, [X[6]] * 13], axis = 0), dtype = torch.float32)

dataset= []
for t in range(2, T-1):     
    c_t = X[t][:, 0:1]
    c_t1  = X[t-1][:, 0:1]   
    c_t2  = X[t-2][:, 0:1]
    extras = X[t][:, 1:4]  
    new_x = torch.cat([c_t, c_t1, c_t2, extras], dim = 1)
    data_t = Data(
        x = new_x,           # shape [num_nodes, num_features]
        edge_index = edge_index,
        y = X[t+1][:, 0]   # next crowding
    )
    dataset.append(data_t)


loader = DataLoader(dataset, batch_size=4, shuffle=True)
"""
print(data_t) # Data(x=[5, 4], edge_index=[2], y=[5])
print(data_t.x.shape)        # should be [5, 4]
print(data_t.edge_index.shape)  # should be [2, E]
print(data_t.y.shape)  
print(data_t.x) # make sure to convert to tensor
print(data_t.y)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CrowdGNN(in_channels=6, hidden_channels=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()

for epoch in range(1, 201):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)              # shape [batch.num_nodes]
        target = batch.y                 # shape [batch.num_nodes]
        loss = loss_fn(pred, target)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(loader):.6f}")

model.eval()
with torch.no_grad():
    pred = model(dataset[-1].to(device))
print(pred[0].item()) # prediction of time t + 1

pred = model(dataset[-1])
# current_crowding = dataset[-1].x[0, 0]
# print(pred[0]) prediction of time t

# Forecast over horizon H

H = 20
preds = []
current_data = dataset[-1]
t0 = T-2

for step in range(H):
    with torch.no_grad():
        pred = model(current_data)
    preds.append(pred)
    next_idx = min(t0 + step + 1, T - 1) # Note for actual, make very long dataset
    extras_next = X[next_idx][:, 1:4]
    c_t = pred.unsqueeze(1)
    c_t1  = current_data.x[:, 0:1]
    c_t2  = current_data.x[:, 1:2]

    new_x = torch.cat([c_t, c_t1, c_t2, extras_next], dim=1)
    current_data = Data(x=new_x, edge_index=edge_index)

print(preds)

