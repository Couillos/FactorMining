from deap import tools


class CrowdedTournamentSelector:
    def __init__(self, tournament_size: int = 2):
        self.tournament_size = tournament_size

    def select(self, individuals, k):
        return tools.selTournamentDCD(individuals, k)
