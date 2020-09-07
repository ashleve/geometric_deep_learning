import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, DataLoader
from models import GCN, GAT, SGCN
from utils import evaluate
import wandb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

DATASET_NAME = "MNISTSuperpixels"
USE_POSITION_FEATURES = True

path = "data/" + DATASET_NAME
if USE_POSITION_FEATURES:
    path += "_with_pos_features"
path += "/"

print(path)
print("Reading dataset...")
train_data = torch.load(path + DATASET_NAME + "_train_data.pt")
test_data = torch.load(path + DATASET_NAME + "_test_data.pt")
print("Train data length:", len(train_data))
print("Test data length:", len(test_data))
print(train_data[0])
print(test_data[0])

job_type = DATASET_NAME
if USE_POSITION_FEATURES:
    job_type += "_with_pos_features"

wandb.init(
    project="geometric_deep_learning_with_superpixels",
    group="GAT",
    job_type=job_type
)


config = wandb.config
config.dataset = DATASET_NAME
config.uses_position_features = USE_POSITION_FEATURES
config.train_data_length = len(train_data)
config.test_data_length = len(test_data)
config.num_node_features = train_data[0]['x'].shape[1]
config.num_classes = 10
config.batch_size = 32
config.learning_rate = 0.001
config.device = device.type


train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=config.batch_size)
del train_data
del test_data


model = GAT(config.num_node_features, config.num_classes)
model = model.to(device)
wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
config.optimizer = optimizer.__class__.__name__


for i in range(100):

    model.train()
    total_train_loss = 0

    for data in train_loader:
        data = data.to(device)
        out = model(data)
        target = data.label.to(device)

        optimizer.zero_grad()
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    total_correct, total_test_loss = evaluate(model, device, test_loader)

    wandb.log({
        "train_loss": total_train_loss / config.train_data_length,
        "test_loss": total_test_loss / config.test_data_length,
        "accuracy": total_correct * 100 / config.test_data_length,
    })

    print(f"Ep: {i}, Acc: {total_correct * 100 / config.test_data_length}")
