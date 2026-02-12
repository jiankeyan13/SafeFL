"""
data 层最小回归测试：锁定 task 生成与 seed 可复现行为。
运行前需先注册假数据集，避免下载真实 CIFAR10。
"""
import os
import tempfile
import unittest

import numpy as np

# 必须在导入 data 业务模块前注册假数据集
import tests.data.fake_dataset  # noqa: F401
from data.task import Task, TaskSet
from data.partitioner import IIDPartitioner, DirichletPartitioner
from data.dataset_store import DatasetStore
from data.task_generator import TaskGenerator


class TestTaskSet(unittest.TestCase):
    """TaskSet 基础行为回归。"""

    def test_add_and_get_task(self):
        ts = TaskSet()
        ts.add_task(Task("client_0", "tag", "train", [1, 2, 3]))
        t = ts.get_task("client_0", "train")
        self.assertEqual(t.owner_id, "client_0")
        self.assertEqual(t.indices, [1, 2, 3])

    def test_get_task_key_error_when_missing(self):
        ts = TaskSet()
        with self.assertRaises(KeyError):
            ts.get_task("client_0", "train")

    def test_has_task_and_try_get_task(self):
        ts = TaskSet()
        ts.add_task(Task("client_0", "tag", "train", [1, 2]))
        self.assertTrue(ts.has_task("client_0", "train"))
        self.assertFalse(ts.has_task("client_0", "test"))
        self.assertIsNone(ts.try_get_task("client_0", "test"))
        self.assertIsNotNone(ts.try_get_task("client_0", "train"))

    def test_multiple_splits_per_owner(self):
        ts = TaskSet()
        ts.add_task(Task("client_0", "tag", "train", [1, 2]))
        ts.add_task(Task("client_0", "tag", "test", [3]))
        self.assertEqual(ts.get_task("client_0", "train").indices, [1, 2])
        self.assertEqual(ts.get_task("client_0", "test").indices, [3])

    def test_list_client_ids_exclude_server(self):
        ts = TaskSet()
        ts.add_task(Task("client_0", "tag", "train", [1]))
        ts.add_task(Task("client_1", "tag", "train", [2]))
        ts.add_task(Task("server", "tag", "proxy", [3]))
        ids = ts.list_client_ids(exclude_server=True)
        self.assertEqual(sorted(ids), ["client_0", "client_1"])

    def test_list_client_ids_include_server(self):
        ts = TaskSet()
        ts.add_task(Task("client_0", "tag", "train", [1]))
        ts.add_task(Task("server", "tag", "proxy", [2]))
        ids = ts.list_client_ids(exclude_server=False)
        self.assertEqual(sorted(ids), ["client_0", "server"])


class TestPartitioner(unittest.TestCase):
    """Partitioner 行为与可复现性。"""

    def _make_store(self, n: int, n_classes: int = 10):
        targets = np.arange(n, dtype=np.int64) % n_classes
        data = np.zeros((n, 32, 32, 3), dtype=np.uint8)
        ds = _SimpleDataset(data, targets)
        return DatasetStore("test", "train", ds)

    def test_iid_reproducible(self):
        store = self._make_store(300)
        p = IIDPartitioner(seed=42)
        t1 = p.partition(store, num_clients=5, split="train")
        t2 = p.partition(store, num_clients=5, split="train")
        idx1 = t1.get_task("client_0", "train").indices
        idx2 = t2.get_task("client_0", "train").indices
        self.assertEqual(idx1, idx2)

    def test_iid_each_client_has_task(self):
        store = self._make_store(100)
        p = IIDPartitioner(seed=42)
        ts = p.partition(store, num_clients=5, split="train")
        for i in range(5):
            task = ts.get_task(f"client_{i}", "train")
            self.assertGreater(len(task.indices), 0)

    def test_dirichlet_each_client_has_task(self):
        store = self._make_store(500, n_classes=10)
        p = DirichletPartitioner(alpha=0.5, seed=42)
        ts = p.partition(store, num_clients=5, split="train")
        for i in range(5):
            task = ts.get_task(f"client_{i}", "train")
            self.assertGreater(len(task.indices), 0)


class TestTaskGenerator(unittest.TestCase):
    """TaskGenerator 核心行为与 seed 可复现性。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_each_client_has_train_task(self):
        from data.partitioner import DirichletPartitioner
        gen = TaskGenerator(
            dataset_name="cifar10",
            root=self.tmp,
            partitioner=DirichletPartitioner(alpha=0.5, seed=42),
            num_clients=5,
            val_ratio=0.2,
            seed=42,
        )
        tasks, stores = gen.generate()
        for i in range(5):
            t = tasks.get_task(f"client_{i}", "train")
            self.assertGreater(len(t.indices), 0)
            self.assertIn("aug", t.dataset_tag)

    def test_val_ratio_positive_has_local_test(self):
        from data.partitioner import DirichletPartitioner
        gen = TaskGenerator(
            dataset_name="cifar10",
            root=self.tmp,
            partitioner=DirichletPartitioner(alpha=0.5, seed=42),
            num_clients=5,
            val_ratio=0.2,
            seed=42,
        )
        tasks, _ = gen.generate()
        t = tasks.get_task("client_0", "test")
        self.assertGreater(len(t.indices), 0)
        self.assertIn("plain", t.dataset_tag)

    def test_server_has_global_test_and_proxy(self):
        from data.partitioner import DirichletPartitioner
        gen = TaskGenerator(
            dataset_name="cifar10",
            root=self.tmp,
            partitioner=DirichletPartitioner(alpha=0.5, seed=42),
            num_clients=5,
            val_ratio=0.2,
            seed=42,
        )
        tasks, _ = gen.generate()
        t_global = tasks.get_task("server", "test_global")
        t_proxy = tasks.get_task("server", "proxy")
        self.assertGreater(len(t_global.indices), 0)
        self.assertGreater(len(t_proxy.indices), 0)

    def test_seed_reproducibility(self):
        from data.partitioner import DirichletPartitioner
        cfg = dict(
            dataset_name="cifar10",
            root=self.tmp,
            partitioner=DirichletPartitioner(alpha=0.5, seed=42),
            num_clients=5,
            val_ratio=0.2,
            seed=42,
        )
        g1 = TaskGenerator(**cfg)
        g2 = TaskGenerator(**cfg)
        t1, _ = g1.generate()
        t2, _ = g2.generate()
        idx1 = t1.get_task("client_0", "train").indices
        idx2 = t2.get_task("client_0", "train").indices
        self.assertEqual(idx1, idx2)


class _SimpleDataset:
    """最小 Dataset，供 Partitioner 测试。"""
    def __init__(self, data, targets):
        self.data = data
        self.targets = np.asarray(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], int(self.targets[i])
