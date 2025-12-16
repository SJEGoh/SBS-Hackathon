from data import get_NS, get_EW, get_NE, get_CC, get_DT, get_TE, get_BP, get_SK, get_PG, return_data, edge_weights, get_stations
from helper import get_edges, generate_extras
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GNN_model import CrowdGNN
import torch
import pandas as pd

phys_edge = (get_edges(get_NS()) + get_edges(get_EW()) + get_edges(get_NE()) + 
                          get_edges(get_CC()) + get_edges(get_DT()) + get_edges(get_TE()) + 
                          get_edges(get_BP()) + get_edges(get_SK()) + get_edges(get_PG()))
edge_index = torch.tensor(phys_edge, dtype = torch.long).T
print(edge_index.shape)
edge_weight = edge_weights(edge_index)
print(edge_index.shape)


X = return_data()
T = 120
N = 182
F = 6
dataset = []
for t in range(2, T-1):
    c_t = X[t][:, 0:1]
    c_t2 = X[t-1][:, 0:1]
    c_t3 = X[t-2][:, 0:1]
    extras_t = X[t][:, 1:6]
    x_t = torch.cat([c_t, c_t2, c_t3, extras_t], dim = 1)
    data_t = Data(
        x=x_t,                    # (N, 6)
        edge_index=edge_index,      # (2, E)    # (E,)
        edge_weight = edge_weight,
        y=X[t+1][:, 0] - X[t][:, 0],             # (N,)
    )
    dataset.append(data_t)

ys = torch.cat([d.y for d in dataset], dim=0)
print("y stats:", ys.mean().item(), ys.std().item(), ys.min().item(), ys.max().item())

split = int(0.8 * len(dataset))
train_ds = dataset[:split]
test_ds  = dataset[split:]

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrowdGNN(in_channels=F+2, hidden_channels=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-4)
loss_fn = torch.nn.MSELoss()

for epoch in range(1, 1001):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)              # shape [batch.num_nodes]
        target = batch.y                 # shape [batch.num_nodes]
        loss = loss_fn(pred, target)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.6f}")


def mse_tensor(a, b):
    return torch.mean((a - b)**2)

# Mean baseline value from TRAIN ONLY (important)
ys_train = torch.cat([d.y for d in train_ds], dim=0)
y_mean_delta = ys_train.mean()


mse_zero = []
mse_mean = []
mse_model = []

model.eval()
with torch.no_grad():
    for d in test_ds:
        y_true = d.y

        y_zero = torch.zeros_like(y_true)
        y_mean = torch.full_like(y_true, y_mean_delta)

        pred_delta = model(d.to(device)).cpu()

        mse_zero.append(mse_tensor(y_zero, y_true).item())
        mse_mean.append(mse_tensor(y_mean, y_true).item())
        mse_model.append(mse_tensor(pred_delta, y_true).item())

print("Zero baseline MSE:", sum(mse_zero)/len(mse_zero))
print("Mean delta MSE:", sum(mse_mean)/len(mse_mean))
print("Model delta MSE:", sum(mse_model)/len(mse_model))


H = 20

current_data = dataset[-1]
t0 = T-2

preds = []
current_data = dataset[-1].to(device)

for step in range(H):
    with torch.no_grad():
        pred = model(current_data)
    preds.append(pred)
    N = current_data.num_nodes
    extras_next = generate_extras(t0 + step + 1, N, device, month=9, is_weekend=0)
    delta = pred.unsqueeze(1)
    log_t  = current_data.x[:, 0:1]            # (N,1) current log flow
    log_t1 = log_t + delta    
    c_t1  = log_t
    c_t2  = current_data.x[:, 1:2]

    new_x = torch.cat([log_t1, c_t1, c_t2, extras_next], dim=1)
    current_data = Data(x=new_x, edge_index=edge_index, edge_weight = edge_weight)

for i in [0, 5, 12]:
    print(i, pred[i])

preds_tensor = torch.stack(preds)  # (H, N)

print(preds_tensor.mean(dim=1))    # avg crowding over time
print(preds_tensor.max(dim=1).values)  # worst station each hour

stations = get_stations()
df_pred = pd.DataFrame({
    "station_id": stations,
    "delta_log_flow": pred.cpu().numpy()
})


