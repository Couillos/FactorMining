# selection.py
"""Selection operators for NSGA-II engine.

Previous dead code removed in T5.8. The engine uses DEAP's built-in
selNSGA2 (which already performs crowded tournament selection via
sort_non_dominated + crowding distance) directly, so no custom
selector class is needed.
"""
