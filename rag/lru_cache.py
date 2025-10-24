from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional

class LRUCache:
    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self._od = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        if key not in self._od:
            return None
        self._od.move_to_end(key)
        return self._od[key]

    def put(self, key: Any, value: Any) -> None:
        self._od[key] = value
        self._od.move_to_end(key)
        if len(self._od) > self.capacity:
            self._od.popitem(last=False)

    def get_many(self, keys: Iterable[Any]) -> tuple[dict, list]:
        found = {}
        misses = []
        for key in keys:
            if key in self._od:
                # move_to_end to mark as used
                self._od.move_to_end(key)
                found[key]=self._od[key]
            else:
                misses.append(key)
        return found, misses

    def put_many(self, items: Dict[Any, Any]) -> None:
        for key, value in items.items():
            self._od[key] = value
            self._od.move_to_end(key)

        # Evict extra items if over capacity
        while len(self._od) > self.capacity:
            self._od.popitem(last=False)
