from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Task:
    owner_id: str
    dataset_tag: str
    split: str
    indices: list[int]


class TaskSet:
    def __init__(self):
        self._tasks: dict[str, dict[str, Task]] = {}

    def add_task(self, task: Task) -> None:
        oid = task.owner_id
        split = task.split
        if oid not in self._tasks:
            self._tasks[oid] = {}
        self._tasks[oid][split] = task

    def get_task(self, oid: str, split: str) -> Task:
        return self._tasks[oid][split]

    def has_task(self, oid: str, split: str) -> bool:
        return oid in self._tasks and split in self._tasks[oid]

    def try_get_task(self, oid: str, split: str) -> Optional[Task]:
        if self.has_task(oid, split):
            return self._tasks[oid][split]
        return None

    def list_client_ids(self, exclude_server: bool = True) -> List[str]:
        """返回所有 owner 的 id 列表。exclude_server=True 时排除 server。"""
        ids = list(self._tasks.keys())
        if exclude_server:
            ids = [cid for cid in ids if cid != "server"]
        return ids

    def __str__(self) -> str:
        return str(self._tasks)