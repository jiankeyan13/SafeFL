from dataclasses import dataclass
from typing import Optional


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

    def __str__(self) -> str:
        return str(self._tasks)