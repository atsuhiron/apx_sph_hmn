import dataclasses


@dataclasses.dataclass
class MNPair:
    m: int
    n: int

    def __hash__(self):
        return hash((self.m, self.n))

    def __str__(self):
        return f"({self.n}, {self.m})"
