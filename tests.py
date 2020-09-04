import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, DataLoader
from models import GCN, GAT, SGCN
from utils import evaluate
import wandb


train_data = torch.load("data/MNISTSuperpixels_train_data.pt")
test_data = torch.load("data/MNISTSuperpixels_test_data.pt")
print(len(train_data))
print(len(test_data))


wandb.init(project="Superpixels_MNIST", group="Simple_GCN", job_type="eval")
config = wandb.config
config.train_data_length = len(train_data)
config.test_data_length = len(test_data)
config.num_node_features = train_data[0]['x'].shape[1]
config.num_classes = 10
config.batch_size = 32
config.learning_rate = 0.001


train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=config.batch_size)
del train_data
del test_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.device = device.type
print(device)

model = GCN(config.num_node_features, config.num_classes)
model = model.to(device)
wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
config.optimizer = optimizer.__class__.__name__


# init log
total_correct, total_test_loss = evaluate(model, device, test_loader)
_, total_train_loss = evaluate(model, device, train_loader)
accuracy = total_correct * 100 / config.test_data_length
train_loss = total_train_loss / config.train_data_length
test_loss = total_test_loss / config.test_data_length
wandb.log({
    "train_loss": train_loss,
    "test_loss": test_loss,
    "accuracy": accuracy,
})

print("Training started...")

for i in range(100):

    model.train()
    total_train_loss = 0

    for data in train_loader:
        data = data.to(device)
        out = model(data)
        target = torch.tensor(data.label).to(device)

        optimizer.zero_grad()
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    total_correct, total_test_loss = evaluate(model, device, test_loader)
    accuracy = total_correct * 100 / config.test_data_length

    train_loss = total_train_loss / config.train_data_length
    test_loss = total_test_loss / config.test_data_length

    wandb.log({
        "train_loss": train_loss,
        "test_loss": test_loss,
        "accuracy": accuracy,
    })

    print(f"Ep: {i}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Acc: {accuracy:.2f}%")
