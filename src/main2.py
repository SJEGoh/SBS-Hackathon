import math
import torch
from helper import month_norm, sincos_hour, is_weekend
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GNN_model import CrowdGNN

# ----- Nodes -----
stations = ["NS23", "CC5", "DT21", "DT31", "NS4/BP1"]
station_to_idx = {s:i for i,s in enumerate(stations)}
N = len(stations)

# ----- Time index -----
time_index = [
  ("2025-09","WEEKDAY",  8),
  ("2025-09","WEEKDAY",  9),
  ("2025-09","WEEKDAY", 18),
  ("2025-09","WEEKDAY", 19),
  ("2025-10","WEEKDAY",  8),
  ("2025-10","WEEKDAY",  9),
  ("2025-10","WEEKENDS/HOLIDAY", 14),
  ("2025-10","WEEKENDS/HOLIDAY", 15),
  ("2025-11","WEEKDAY",  8),
  ("2025-11","WEEKDAY",  9),
  ("2025-12","WEEKENDS/HOLIDAY", 14),
  ("2025-12","WEEKENDS/HOLIDAY", 15),
]
T = len(time_index)

# ----- Edges from OD trips -----
edges_named = [
  ("NS23","CC5"),
  ("DT21","DT31"),
  ("NS4/BP1","CC5"),
  ("CC5","DT21"),
  ("DT31","NS23"),
]
edge_index = torch.tensor([
  [station_to_idx[a] for a,b in edges_named],
  [station_to_idx[b] for a,b in edges_named],
], dtype=torch.long)

edge_weight = torch.tensor([0.15, 0.90, 0.60, 0.35, 0.20], dtype=torch.float32)
# ----- Dummy "crowding" built from (tap in/out) style patterns -----
# We'll just craft plausible station-specific patterns with peak hours + weekend effects.
base = torch.tensor([0.55, 0.65, 0.60, 0.58, 0.50])  # per station baseline

X = torch.zeros((T, N, 6), dtype=torch.float32)  # [crowding, sin, cos, weekend, month_norm, discount]

for t, (ym, day_type, hour) in enumerate(time_index):
    s, c = sincos_hour(hour)
    wk = is_weekend(day_type)
    mn = month_norm(ym)

    # create a peak-ish effect around 8-9 and 18-19
    peak = 0.15 if hour in (8, 9, 18, 19) and wk == 0 else 0.05 if wk == 1 else 0.0
    season = 0.03 * mn  # slightly higher toward Dec

    crowd = base + peak + season
    # add small station differences (CBD-ish stations)
    crowd[station_to_idx["CC5"]] += 0.05
    crowd[station_to_idx["DT31"]] += 0.03

    # policy knob: discount active only on weekend afternoons in this dummy
    discount = 1.0 if (wk == 1.0 and hour in (14, 15)) else 0.0

    X[t, :, 0] = torch.clamp(crowd, 0.0, 1.0)
    X[t, :, 1] = s
    X[t, :, 2] = c
    X[t, :, 3] = wk
    X[t, :, 4] = mn
    X[t, :, 5] = discount




# ----- Build PyG dataset: predict next-step crowding -----
dataset = []
for t in range(2, T-1):
    c_t = X[t][:, 0:1]
    c_t2 = X[t-1][:, 0:1]
    c_t3 = X[t-3][:, 0:1]
    extras_t = X[t][:, 1:6]
    x_t = torch.cat([c_t, c_t2, c_t3, extras_t], dim = 1)
    data_t = Data(
        x=x_t,                    # (N, 6)
        edge_index=edge_index,      # (2, E)
        edge_weight=edge_weight,    # (E,)
        y=X[t+1][:, 0],             # (N,)
    )
    dataset.append(data_t)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print("X shape:", X.shape)                 # (T, N, 6)
print("edge_index:", edge_index.shape)     # (2, E)
print("edge_weight:", edge_weight.shape)   # (E,)
print("sample:", dataset[0])
print("sample x:", dataset[0].x.shape, "y:", dataset[0].y.shape)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CrowdGNN(in_channels=8, hidden_channels=16).to(device)
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

# current_crowding = dataset[-1].x[0, 0]
# print(pred[0]) prediction of time t

# Forecast over horizon H

H = 20

current_data = dataset[-1]
t0 = T-2

preds = []
current_data = dataset[-1].to(device)

ym0, day_type0, hour0 = time_index[t0]
mn0 = month_norm(ym0)
wk0 = is_weekend(day_type0)

for step in range(H):
    with torch.no_grad():
        pred = model(current_data)
    preds.append(pred)
    next_idx = min(t0 + step + 1, T - 1) # Note for actual, make very long dataset
    extras_next = X[next_idx][:, 1:6]
    c_t = pred.unsqueeze(1)
    c_t1  = current_data.x[:, 0:1]
    c_t2  = current_data.x[:, 1:2]

    new_x = torch.cat([c_t, c_t1, c_t2, extras_next], dim=1)
    current_data = Data(x=new_x, edge_index=edge_index, edge_weight = edge_weight)

print(f"Preds:", preds)

