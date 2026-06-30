from dataclasses import dataclass
from deap import gp


@dataclass
class TreeChromosome:
    tree: gp.PrimitiveTree
    fitness: tuple[float, ...] | None = None
