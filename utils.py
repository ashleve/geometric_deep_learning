import numpy as np
from skimage.segmentation import slic, mark_boundaries
import networkx as nx
import matplotlib.pyplot as plt
NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

NUM_FEATURES = 3
NUM_CLASSES = 10


def get_graph_from_image(image, desired_nodes=75):
    # load the image and convert it to a floating point data type
    segments = slic(image, n_segments=desired_nodes, slic_zero=True)

    asegments = np.array(segments)

    num_nodes = np.max(asegments)
    nodes = {
        node: {
            "rgb_list": [],
            "pos_list": []
        } for node in range(num_nodes + 1)
    }

    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            node = asegments[y, x]
            rgb = image[y, x, :]
            pos = np.array([float(x) / width, float(y) / height])
            nodes[node]["rgb_list"].append(rgb)
            nodes[node]["pos_list"].append(pos)
        # end for
    # end for

    G = nx.Graph()

    for node in nodes:
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        # rgb
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        # rgb_std = np.std(nodes[node]["rgb_list"], axis=0)
        # rgb_gram = np.matmul( nodes[node]["rgb_list"].T, nodes[node]["rgb_list"] ) / nodes[node]["rgb_list"].shape[0]
        # Pos
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        # pos_std = np.std(nodes[node]["pos_list"], axis=0)
        # pos_gram = np.matmul( nodes[node]["pos_list"].T, nodes[node]["pos_list"] ) / nodes[node]["pos_list"].shape[0]
        # Debug

        features = np.concatenate(
            [
                np.reshape(rgb_mean, -1),
                # np.reshape(rgb_std, -1),
                # np.reshape(rgb_gram, -1),
                np.reshape(pos_mean, -1),
                # np.reshape(pos_std, -1),
                # np.reshape(pos_gram, -1)
            ]
        )
        G.add_node(node, features=list(features))
    # end

    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    # Adjacency loops
    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])

    # Self loops
    for node in nodes:
        G.add_edge(node, node)

    n = len(G.nodes)
    m = len(G.edges)
    h = np.zeros([n, NUM_FEATURES]).astype(NP_TORCH_FLOAT_DTYPE)
    edges = np.zeros([2 * m, 2]).astype(NP_TORCH_LONG_DTYPE)
    for e, (s, t) in enumerate(G.edges):
        edges[e, 0] = s
        edges[e, 1] = t

        edges[m + e, 0] = t
        edges[m + e, 1] = s
    # end for
    for i in G.nodes:
        h[i, :] = G.nodes[i]["features"]
    # end for
    del G
    return h, edges