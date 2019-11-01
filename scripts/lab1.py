from collections import namedtuple

from docplex.mp.model import Model
from docplex.util.environment import get_environment
import numpy as np

"""
3 types of planes,
4 routes

"""

PLANES = [
    ("Plane1", 50, 20, [3, 2, 2, 1], [1000, 1100, 1200, 1500]),
    ("Plane2", 30, 22, [4, 3, 3, 2], [800, 900, 1000, 1000]),
    ("Plane3", 20, 24, [5, 5, 4, 2], [600, 800, 800, 900]),
]

routes = [1, 2, 3, 4]

# Total num of passengers daily
passengers_en_route = [1000, 2000, 900, 1200]

# For empty seat
penalty = [40, 50, 45, 70]

Plane = namedtuple("Plane", [
                   "type", "capasity", "num_of_aircrafts", "daily_trips_en_route", "operational_cost"])


def build_airplane_model(**kwargs):
    planes = [Plane(*p) for p in PLANES]

    mdl = Model(name="planes", **kwargs)
    # decision_vars
    nb_planes = len(planes)
    nb_routes = len(routes)

    optim_planes = mdl.continuous_var_matrix(
        nb_planes, nb_routes, name='optim_planes')

    empty_seats = mdl.continuous_var_list(routes, name="empty_seats")
    # constraints
    mdl.clear_constraints()
    mdl.add_constraint(mdl.sum(optim_planes[0, i]
                               for i in range(nb_routes)) <= 7)
    mdl.add_constraint(mdl.sum(optim_planes[1, i]
                               for i in range(nb_routes)) <= 15)
    mdl.add_constraint(mdl.sum(optim_planes[2, i]
                               for i in range(nb_routes)) <= 2)

    mdl.add_constraint(mdl.sum(planes[i].capasity * planes[i].daily_trips_en_route[0]
                               * optim_planes[i, 0] for i in range(3)) + empty_seats[0] == passengers_en_route[0])
    mdl.add_constraint(mdl.sum(planes[i].capasity * planes[i].daily_trips_en_route[1]
                               * optim_planes[i, 1] for i in range(3)) + empty_seats[1] == passengers_en_route[1])
    mdl.add_constraint(mdl.sum(planes[i].capasity * planes[i].daily_trips_en_route[2]
                               * optim_planes[i, 2] for i in range(3)) + empty_seats[2] == passengers_en_route[2])
    mdl.add_constraint(mdl.sum(planes[i].capasity * planes[i].daily_trips_en_route[3]
                               * optim_planes[i, 3] for i in range(3)) + empty_seats[3] == passengers_en_route[3])

    mdl.minimize(mdl.sum(
        planes[i].operational_cost[j] * planes[i].daily_trips_en_route[j] * optim_planes[i, j] for i in range(3) for j in range(4))
        + mdl.sum(penalty[j]*empty_seats[j] for j in range(nb_routes))
    )

    return mdl


if __name__ == '__main__':
    mdl = build_airplane_model()
    mdl.print_information()

    solution = mdl.solve(log_output=True)
    mdl.float_precision = 3

    for var, value in solution.iter_var_values():
        print("{}: {}".format(var, value))

    print("dual values for model constraints:")
    for ctn, dual in zip(mdl.iter_constraints(), mdl.dual_values(mdl.iter_constraints())):
        print("{} : {}".format(ctn, dual))

    slack_values = mdl.slack_values(mdl.iter_constraints())
    print("slack values: {}".format(slack_values))
    print("deficit values for constraints:")
    deficit_count = 0
    for cnt, slack_value in zip(mdl.iter_constraints(), slack_values):
        if slack_value != 0:
            continue
        deficit_count += 1
        print("{}) {}".format(deficit_count, cnt))
    mdl.get_solve_status()
    solution.display()
