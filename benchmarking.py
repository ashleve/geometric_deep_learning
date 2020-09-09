import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.datasets import MNISTSuperpixels
from models import GCN, GAT, SGCN
from utils import evaluate
import wandb


DATASET_NAME = "MNISTSuperpixels"
USE_POSITION_FEATURES = False
USE_AUGMENTATION = False
MODEL = GCN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if USE_POSITION_FEATURES:
    DATASET_NAME += "_with_pos_features"
path = "data/" + DATASET_NAME + "/"
if USE_AUGMENTATION:
    DATASET_NAME += "_augmented"
model_name = MODEL.__name__
print("Dataset:", DATASET_NAME)
print("Model:", model_name)

print("Reading dataset...")
train_data = torch.load(path + "train_data.pt")
test_data = torch.load(path + "test_data.pt")

# train_data = MNISTSuperpixels("data/mnist/", True, transform=T.Cartesian())
# test_data = MNISTSuperpixels("data/mnist/", False, transform=T.Cartesian())

print("Train data length:", len(train_data))
print("Test data length:", len(test_data))
print(train_data[0])
print(test_data[0])


wandb.init(
    project="geometric_deep_learning_with_superpixels",
    group=model_name,
    job_type=DATASET_NAME
)
print("Initialized wandb")

config = wandb.config
config.dataset = DATASET_NAME
config.uses_position_features = USE_POSITION_FEATURES
config.use_augmentation = USE_AUGMENTATION
config.model = model_name
config.train_data_length = len(train_data)
config.test_data_length = len(test_data)
config.num_node_features = train_data[0]['x'].shape[1]
config.num_classes = 10
config.batch_size = 64
config.learning_rate = 0.001
config.device = device.type



train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=config.batch_size)
print(train_data[0].pos.shape)
print(train_data[0].x.shape)
del train_data
del test_data


model = MODEL(config)
model = model.to(device)
wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
config.optimizer = optimizer.__class__.__name__


rotation_0 = T.RandomRotate(degrees=180, axis=0)
rotation_1 = T.RandomRotate(degrees=180, axis=1)
rotation_2 = T.RandomRotate(degrees=180, axis=2)


def train():
	best_acc = 0
	for i in range(100):

	    model.train()
	    total_train_loss = 0

	    for data in train_loader:
	        data = data.to(device)
	        # print("-------")
	        # print(data.pos)

	        # if config.use_augmentation:
	        #     data = rotation_0(data)
	            # data = rotation_1(data)
	            # data = rotation_2(data)
	        # print(data.pos)
	        # print("-------")

	        out = model(data)

	        optimizer.zero_grad()
	        loss = F.nll_loss(out, data.label)
	        loss.backward()
	        optimizer.step()

	        total_train_loss += loss.item()

	    total_correct, total_test_loss = evaluate(model, device, test_loader)

	    train_loss = total_train_loss / config.train_data_length
	    test_loss = total_test_loss / config.test_data_length
	    acc = total_correct * 100 / config.test_data_length
	    if acc > best_acc:
	        best_acc = acc

	    wandb.log({
	        "train_loss": train_loss,
	        "test_loss": test_loss,
	        "accuracy": acc,
	        "best_accuracy": best_acc
	    })

	    print(f"Ep: {i}, acc: {acc}, best acc: {best_acc}")


train()