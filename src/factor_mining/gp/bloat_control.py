from deap import gp


def static_limit_factory(max_nodes: int = 17):
    def limit(ind):
        return len(ind) <= max_nodes
    return limit


def double_tournament_selector(toolbox, individuals, k, parsimony: float = 1.4):
    return gp.selDoubleTournament(
        individuals=individuals,
        k=k,
        fitness_size=2,
        parsimony_size=2 if parsimony > 0 else 0,
        parsimony_factor=parsimony,
    )
