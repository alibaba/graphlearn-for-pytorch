import base64
import json
import random
import unittest
from typing import List, Optional

import graphscope.learning.graphlearn_torch as glt
import torch
from dist_test_utils import *
from dist_test_utils import _prepare_hetero_dataset
from graphscope.learning.gs_feature_store import GsFeatureStore
from graphscope.learning.gs_graph_store import GsGraphStore
from parameterized import parameterized
from torch_geometric.data import EdgeAttr, TensorAttr
from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.loader import NeighborLoader


def run_test_as_server(
    num_servers: int,
    num_clients: int,
    server_rank: List[int],
    master_port: int,
    dataset: glt.distributed.DistDataset,
    is_dynamic: bool = False,
):
    print(f"[Server {server_rank}] Initializing server ...")
    glt.distributed.init_server(
        num_servers=num_servers,
        num_clients=num_clients,
        server_rank=server_rank,
        dataset=dataset,
        master_addr="localhost",
        master_port=master_port,
        request_timeout=30,
        num_rpc_threads=2,
        server_group_name="pyg_remote_backend_test_server",
        is_dynamic=is_dynamic,
    )

    print(f"[Server {server_rank}] Waiting for exit ...")
    glt.distributed.wait_and_shutdown_server()

    print(f"[Server {server_rank}] Exited ...")


def run_test_as_client(
    num_servers: int,
    num_clients: int,
    client_rank: int,
    master_port: int,
    input_nodes,
    input_edges,
    feature_size,
    check_fn,
    edge_dir="out",
    is_dynamic: bool = False,
):
    print(f"[Client {client_rank}] Initializing client ...")
    glt.distributed.init_client(
        num_servers=num_servers,
        num_clients=num_clients,
        client_rank=client_rank,
        master_addr="localhost",
        master_port=master_port,
        num_rpc_threads=1,
        client_group_name="pyg_remote_backend_test_client",
        is_dynamic=is_dynamic,
    )

    print(f"[Client {client_rank}] Check function {check_fn.__name__} ...")
    config = base64.b64encode(
        json.dumps(
            {
                "node_features": {
                    input_nodes[i]: [f"feat_{j}" for j in range(feature_size[i])]
                    for i in range(len(input_nodes))
                },
                "node_labels": {
                    input_nodes[i]: "label" for i in range(len(input_nodes))
                },
                "edges": input_edges,
                "edge_dir": edge_dir,
            }
        ).encode("utf-8")
    ).decode("utf-8")

    feature_store = GsFeatureStore(config)
    graph_store = GsGraphStore(config)

    check_fn(feature_store, graph_store, input_nodes, input_edges, feature_size)

    print(f"[Client {client_rank}] Shutdowning ...")
    glt.distributed.shutdown_client()

    print(f"[Client {client_rank}] Exited ...")


def _check_feature_store(
    feature_store, graph_store, input_nodes, input_edges, feature_size
):
    tc = unittest.TestCase()
    for i in range(len(input_nodes)):
        feature = feature_store.get_tensor(
            TensorAttr(group_name=input_nodes[i], attr_name="x", index=0)
        )
        tc.assertEqual(feature.shape[-1], feature_size[i])
        label = feature_store.get_tensor(
            TensorAttr(group_name=input_nodes[i], attr_name="label", index=0)
        )
        tc.assertEqual(label.shape[-1], 1)
        size = feature_store.get_tensor_size(TensorAttr(group_name=input_nodes[i]))
        tc.assertEqual(size[0], feature_size[i])
    tensor_attrs = feature_store.get_all_tensor_attrs()
    tc.assertEqual(len(tensor_attrs), len(input_nodes) * 2)
    for tensor_attr in tensor_attrs:
        tc.assertIn(tensor_attr.group_name, input_nodes)
        tc.assertIn(tensor_attr.attr_name, ["x", "label"])


def _check_graph_store(
    feature_store, graph_store, input_nodes, input_edges, feature_size
):
    tc = unittest.TestCase()
    edge_attrs = graph_store.get_all_edge_attrs()
    tc.assertEqual(len(edge_attrs), len(input_edges))
    for index, edge_attr in enumerate(edge_attrs):
        tc.assertEqual(edge_attr.edge_type, input_edges[index])
        tc.assertEqual(edge_attr.layout, EdgeLayout.CSR)
        tc.assertEqual(edge_attr.is_sorted, False)
        tc.assertEqual(len(edge_attr.size), 2)
    for i in range(len(input_edges)):
        edge_index = graph_store.get_edge_index(EdgeAttr(input_edges[i], "csr"))
        tc.assertEqual(len(edge_index), 2)
        tc.assertEqual(len(edge_index[0]), edge_attrs[i].size[0] + 1)


def _check_pyg_neighbor_loader_with_pyg_remote_backend(
    feature_store,
    graph_store,
    input_nodes,
    input_edges,
    feature_size,
    num_neighbors=[4, 3, 2],
    loader_batch_size=4,
):
    tc = unittest.TestCase()
    sample_node_type = input_nodes[1]
    loader = NeighborLoader(
        data=(feature_store, graph_store),
        batch_size=loader_batch_size,
        num_neighbors={input_edges[i]: num_neighbors for i in range(len(input_edges))},
        shuffle=False,
        input_nodes=sample_node_type,
    )

    for batch in loader:
        batch_node = batch[sample_node_type]
        random_choose = random.randint(0, batch_node.batch_size - 1)
        node_id = batch_node.n_id[random_choose].item()
        feature_result = feature_store.get_tensor(
            TensorAttr(group_name=sample_node_type, attr_name="x", index=node_id)
        )[0]
        tc.assertTrue(torch.equal(batch_node.x[random_choose], feature_result))

        for i in range(len(input_edges)):
            batch_edge = batch[input_edges[i]]
            edge_result = graph_store.get_edge_index(EdgeAttr(input_edges[i], "csc"))
            indices_start = edge_result[1][node_id]
            indices_end = edge_result[1][node_id + 1]
            if indices_start == indices_end:
                continue
            neighbors1 = edge_result[0][indices_start:indices_end]

            row_ = (batch_edge.edge_index[1] == random_choose).nonzero(as_tuple=False)
            col_start = row_[0].item()
            col_end = row_[-1].item()
            col = batch_edge.edge_index[0][col_start : col_end + 1]
            neighbors2 = batch_node.n_id[col]

            tc.assertTrue(torch.isin(neighbors2, neighbors1).all().item())


class PygRemoteBackendTestCase(unittest.TestCase):
    def setUp(self):
        self.loader_batch_size = 4
        self.num_neighbors = [4, 3, 2]
        self.dataset0 = _prepare_hetero_dataset(rank=0, edge_dir="out")
        self.dataset1 = _prepare_hetero_dataset(rank=1, edge_dir="out")
        self.master_port = glt.utils.get_free_port()

    @parameterized.expand(
        [
            (1, 2, [user_ntype, item_ntype], [i2i_etype], [512, 256]),
        ]
    )
    def test_feature_store(
        self, num_clients, num_servers, input_nodes, input_edges, feature_size
    ):
        print("\n--- Feature Store Test (server-client mode, remote) ---")
        print(f"--- num_clients: {num_clients} num_servers: {num_servers} ---")

        self.dataset_list = [self.dataset0, self.dataset1]
        # self.input_nodes_list = [self.input_nodes0, self.input_nodes1]

        mp_context = torch.multiprocessing.get_context("spawn")

        server_procs = []
        for server_rank in range(num_servers):
            server_procs.append(
                mp_context.Process(
                    target=run_test_as_server,
                    args=(
                        num_servers,
                        num_clients,
                        server_rank,
                        self.master_port,
                        self.dataset_list[server_rank],
                    ),
                )
            )

        client_procs = []
        for client_rank in range(num_clients):
            client_procs.append(
                mp_context.Process(
                    target=run_test_as_client,
                    args=(
                        num_servers,
                        num_clients,
                        client_rank,
                        self.master_port,
                        input_nodes,
                        input_edges,
                        feature_size,
                        _check_feature_store,
                    ),
                )
            )
        for sproc in server_procs:
            sproc.start()
        for cproc in client_procs:
            cproc.start()

        for sproc in server_procs:
            sproc.join()
        for cproc in client_procs:
            cproc.join()

    @parameterized.expand(
        [
            (1, 2, [user_ntype, item_ntype], [i2i_etype], [512, 256]),
        ]
    )
    def test_graph_store(
        self, num_clients, num_servers, input_nodes, input_edges, feature_size
    ):
        print("\n--- Graph Store Test (server-client mode, remote) ---")
        print(f"--- num_clients: {num_clients} num_servers: {num_servers} ---")

        self.dataset_list = [self.dataset0, self.dataset1]
        # self.input_nodes_list = [self.input_nodes0, self.input_nodes1]

        mp_context = torch.multiprocessing.get_context("spawn")

        server_procs = []
        for server_rank in range(num_servers):
            server_procs.append(
                mp_context.Process(
                    target=run_test_as_server,
                    args=(
                        num_servers,
                        num_clients,
                        server_rank,
                        self.master_port,
                        self.dataset_list[server_rank],
                    ),
                )
            )

        client_procs = []
        for client_rank in range(num_clients):
            client_procs.append(
                mp_context.Process(
                    target=run_test_as_client,
                    args=(
                        num_servers,
                        num_clients,
                        client_rank,
                        self.master_port,
                        input_nodes,
                        input_edges,
                        feature_size,
                        _check_graph_store,
                    ),
                )
            )
        for sproc in server_procs:
            sproc.start()
        for cproc in client_procs:
            cproc.start()

        for sproc in server_procs:
            sproc.join()
        for cproc in client_procs:
            cproc.join()

    @parameterized.expand(
        [
            (1, 2, [user_ntype, item_ntype], [i2i_etype], [512, 256]),
        ]
    )
    def test_pyg_neighbor_loader(
        self, num_clients, num_servers, input_nodes, input_edges, feature_size
    ):
        print(
            "\n--- PyG NeighborLoader with remotebackend Test (server-client mode, remote) ---"
        )
        print(f"--- num_clients: {num_clients} num_servers: {num_servers} ---")

        self.dataset_list = [self.dataset0, self.dataset1]

        mp_context = torch.multiprocessing.get_context("spawn")

        server_procs = []
        for server_rank in range(num_servers):
            server_procs.append(
                mp_context.Process(
                    target=run_test_as_server,
                    args=(
                        num_servers,
                        num_clients,
                        server_rank,
                        self.master_port,
                        self.dataset_list[server_rank],
                    ),
                )
            )

        client_procs = []
        for client_rank in range(num_clients):
            client_procs.append(
                mp_context.Process(
                    target=run_test_as_client,
                    args=(
                        num_servers,
                        num_clients,
                        client_rank,
                        self.master_port,
                        input_nodes,
                        input_edges,
                        feature_size,
                        _check_pyg_neighbor_loader_with_pyg_remote_backend,
                    ),
                )
            )
        for sproc in server_procs:
            sproc.start()
        for cproc in client_procs:
            cproc.start()

        for sproc in server_procs:
            sproc.join()
        for cproc in client_procs:
            cproc.join()


if __name__ == "__main__":
    unittest.main()
