import hashlib
from collections import OrderedDict


class SubtreeCache:
    def __init__(self, maxsize: int = 10000):
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self.maxsize = maxsize

    def _key(self, tree) -> str:
        return hashlib.sha256(str(tree).encode()).hexdigest()

    def get(self, tree) -> tuple[float, ...] | None:
        key = self._key(tree)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, tree, fitness: tuple[float, ...]) -> None:
        key = self._key(tree)
        self._cache[key] = fitness
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
