import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic, mark_boundaries
import networkx as nx


def get_graph_from_image(image, desired_nodes=75, add_position_to_features=True):
    height = image.shape[0]
    width = image.shape[1]
    num_of_features = image.shape[2] + 2 if add_position_to_features else image.shape[2]

    segments = slic(image, n_segments=desired_nodes, slic_zero=True)

    num_of_nodes = np.max(segments) + 1
    nodes = {
        node: {
            "rgb_list": [],
            "pos_list": []
        } for node in range(num_of_nodes)
    }

    # get rgb
    for y in range(height):
        for x in range(width):
            node = segments[y, x]

            rgb = image[y, x, :]
            nodes[node]["rgb_list"].append(rgb)

            pos = np.array([float(x) / width, float(y) / height])
            nodes[node]["pos_list"].append(pos)

    # compute features (from rgb only)
    G = nx.Graph()
    for node in nodes:
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        if add_position_to_features:
            features = np.concatenate((rgb_mean, pos_mean))
        else:
            features = rgb_mean
        G.add_node(node, features=list(features))

    # compute node positions
    segments_ids = np.unique(segments)
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
    centers = centers.astype(int)

    # add edges
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])

    # add self loops
    for node in nodes:
        G.add_edge(node, node)

    # get edge_index
    m = len(G.edges)
    edge_index = np.zeros([2 * m, 2]).astype(np.int64)
    for e, (s, t) in enumerate(G.edges):
        edge_index[e, 0] = s
        edge_index[e, 1] = t
        edge_index[m + e, 0] = t
        edge_index[m + e, 1] = s

    # get features
    x = np.zeros([num_of_nodes, num_of_features]).astype(np.float32)
    for node in G.nodes:
        x[node, :] = G.nodes[node]["features"]

    return x, edge_index, centers


def evaluate(model, device, test_loader):
    model.eval()
    total_correct = 0
    loss = 0

    for data in test_loader:
        data = data.to(device)
        out = model(data)
        _, predicts = out.max(dim=1)
        target = data.label.to(device)
        loss += F.nll_loss(out, target).item()
        correct = predicts.eq(target).sum().item()
        total_correct += correct

    return total_correct, loss
