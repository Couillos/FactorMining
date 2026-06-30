from deap import tools


class ParetoFrontManager:
    def __init__(self):
        self.front = tools.ParetoFront()

    def update(self, individuals):
        self.front.update(individuals)


class EliteScheme:
    def __init__(self, elite_ratio: float = 0.10):
        self.elite_ratio = elite_ratio

    def preserve(self, combined_pop):
        combined_pop.sort(key=lambda ind: ind.fitness.wvalues[0], reverse=True)
        n_elite = max(1, int(len(combined_pop) * self.elite_ratio))
        return combined_pop[:n_elite]
