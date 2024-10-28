import unittest
from collections import defaultdict
from typing import List

import graphlearn_torch as glt
import torch
from dist_test_utils import *
from dist_test_utils import _prepare_hetero_dataset
from graphlearn_torch.distributed.dist_client import request_server
from graphlearn_torch.distributed.dist_server import DistServer
from parameterized import parameterized
from torch_geometric.utils.sparse import index2ptr, ptr2index


def run_test_as_server(
    num_servers: int,
    num_clients: int,
    server_rank: List[int],
    master_port: int,
    dataset: glt.distributed.DistDataset,
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
    )

    print(f"[Server {server_rank}] Waiting for exit ...")
    glt.distributed.wait_and_shutdown_server()

    print(f"[Server {server_rank}] Exited ...")


def run_test_as_client(
    num_servers: int,
    num_clients: int,
    client_rank: int,
    master_port: int,
    node_type,
    node_index,
    feature_size,
    edge_type,
    edge_layout,
    check_fn,
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
    )

    print(f"[Client {client_rank}] Check function {check_fn.__name__} ...")

    check_fn(node_type, node_index, feature_size, edge_type, edge_layout)

    print(f"[Client {client_rank}] Shutdowning ...")
    glt.distributed.shutdown_client()

    print(f"[Client {client_rank}] Exited ...")


def _check_feature_store(node_type, node_index, feature_size, edge_type, edge_layout):
    tc = unittest.TestCase()
    patition_ids = request_server(
        0, DistServer.get_node_partition_id, node_type, node_index
    )
    tc.assertTrue(torch.equal(patition_ids, node_index % 2))

    partition_to_ids = defaultdict(list)
    partition_to_indices = defaultdict(list)
    for idx, id in enumerate(node_index):
        partition_id = patition_ids[idx].item()
        partition_to_ids[partition_id].append(id.item())
        partition_to_indices[partition_id].append(idx)

    partition_to_features = defaultdict(list)
    partition_to_labels = defaultdict(list)
    for partition_id, ids in partition_to_ids.items():
        feature = request_server(
            partition_id, DistServer.get_node_feature, node_type, torch.tensor(ids)
        )
        label = request_server(
            partition_id, DistServer.get_node_label, node_type, torch.tensor(ids)
        )
        partition_to_features[partition_id] = feature
        partition_to_labels[partition_id] = label

    node_features = torch.zeros(
        (len(node_index), list(partition_to_features.values())[0].shape[-1])
    )
    node_labels = torch.zeros((len(node_index), 1))
    for partition_id, indices in partition_to_indices.items():
        partition_features = partition_to_features[partition_id]
        partition_labels = partition_to_labels[partition_id]
        for i, idx in enumerate(indices):
            node_features[idx] = partition_features[i]
            node_labels[idx] = partition_labels[i]

    for id, index in enumerate(node_index):
        if index % 2 == 0:
            tc.assertTrue(torch.equal(node_features[id], torch.zeros(feature_size)))
        else:
            if node_type == user_ntype:
                tc.assertTrue(torch.equal(node_features[id], torch.ones(feature_size)))
            else:
                tc.assertTrue(
                    torch.equal(node_features[id], torch.full((feature_size,), 2))
                )
        tc.assertEqual(node_labels[id], index)


def _check_graph_store(node_type, node_index, feature_size, edge_type, edge_layout):
    tc = unittest.TestCase()

    if edge_type == u2i_etype:
        step = 1
    else:
        step = 2

    for server_id in range(2):
        true_rows = []
        true_cols = []
        for v in range(server_id, vnum_total, 2):
            true_rows.extend([v for _ in range(degree)])
            true_cols.extend(
                sorted([((v + i + step) % vnum_total) for i in range(degree)])
            )
        true_rows = torch.tensor(true_rows)
        true_cols = torch.tensor(true_cols)

        (row, col) = request_server(
            server_id, DistServer.get_edge_index, edge_type, edge_layout
        )

        tc.assertTrue(torch.equal(row, true_rows))
        tc.assertTrue(torch.equal(col, true_cols))


class PygRemoteBackendTestCase(unittest.TestCase):
    def setUp(self):
        self.loader_batch_size = 4
        self.num_neighbors = [4, 3, 2]
        self.dataset0 = _prepare_hetero_dataset(rank=0, edge_dir="out")
        self.dataset1 = _prepare_hetero_dataset(rank=1, edge_dir="out")
        self.master_port = glt.utils.get_free_port()

    @parameterized.expand(
        [
            (1, 2, user_ntype, torch.tensor([0]), 512),
            (1, 2, user_ntype, torch.tensor([0, 1, 2, 3]), 512),
            (1, 2, item_ntype, torch.tensor([0]), 256),
            (1, 2, item_ntype, torch.tensor([4, 5, 6, 7]), 256),
        ]
    )
    def test_dist_server_supported_feature_store(
        self, num_clients, num_servers, node_type, node_index, feature_size
    ):
        print(
            "\n--- Function in DistServer supported PyG Remote Backend Test (server-client mode, remote) ---"
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
                        node_type,
                        node_index,
                        feature_size,
                        None,
                        None,
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
            (1, 2, i2i_etype, "coo"),
            (1, 2, u2i_etype, "coo"),
        ]
    )
    def test_dist_server_supported_graph_store(
        self, num_clients, num_servers, edge_type, edge_layout
    ):
        print(
            "\n--- Function in DistServer supported PyG Remote Backend Test (server-client mode, remote) ---"
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
                        None,
                        None,
                        None,
                        edge_type,
                        edge_layout,
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


if __name__ == "__main__":
    unittest.main()
