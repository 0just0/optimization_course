import copy
import time
from collections import namedtuple

import numpy as np
from docplex.mp.model import Model

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

planes = [Plane(*p) for p in PLANES]

mdl = Model(name="planes")

# decision_vars
nb_planes = len(planes)
nb_routes = len(routes)

optim_planes = mdl.continuous_var_matrix(
    nb_planes, nb_routes, name='optim_planes')

empty_seats = mdl.continuous_var_list(routes, name="empty_seats")

# objective function
mdl.minimize(mdl.sum(
    planes[i].operational_cost[j] * planes[i].daily_trips_en_route[j] * optim_planes[i, j] for i in range(3) for j in range(4))
    + mdl.sum(penalty[j]*empty_seats[j] for j in range(nb_routes))
)


def add_constraints(model):
    model.clear_constraints()
    model.add_constraint(model.sum(optim_planes[0, i]
                                   for i in range(nb_routes)) <= 7)
    model.add_constraint(model.sum(optim_planes[1, i]
                                   for i in range(nb_routes)) <= 15)
    model.add_constraint(model.sum(optim_planes[2, i]
                                   for i in range(nb_routes)) <= 2)

    model.add_constraint(model.sum(planes[i].capasity * planes[i].daily_trips_en_route[0]
                                   * optim_planes[i, 0] for i in range(3)) + empty_seats[0] == passengers_en_route[0])
    model.add_constraint(model.sum(planes[i].capasity * planes[i].daily_trips_en_route[1]
                                   * optim_planes[i, 1] for i in range(3)) + empty_seats[1] == passengers_en_route[1])
    model.add_constraint(model.sum(planes[i].capasity * planes[i].daily_trips_en_route[2]
                                   * optim_planes[i, 2] for i in range(3)) + empty_seats[2] == passengers_en_route[2])
    model.add_constraint(model.sum(planes[i].capasity * planes[i].daily_trips_en_route[3]
                                   * optim_planes[i, 3] for i in range(3)) + empty_seats[3] == passengers_en_route[3])

    return model


class BnBNode:
    def __init__(self, mdl, upper_bound=0, hot_start=True):
        self.mdl = mdl
        self.objective_value = upper_bound if hot_start else self.get_initial_bound()
        self.optimal_mdl_solution = None
        self.left_child = None
        self.right_child = None
        self.mdl_solution = None

    def solve(self):
        global UPPER_BOUND
        self.mdl_solution = self.mdl.solve(log_output=False)
        if self.mdl_solution:
            current = self.mdl_solution.objective_value
            print("CURRENT ", current)
            is_int = self.check_is_integer()
            print(is_int)
            if is_int:
                if current < UPPER_BOUND:
                    UPPER_BOUND = current
                    self.optimal_mdl_solution = self.mdl_solution
                    return self.optimal_mdl_solution
                else:
                    return False
            else:
                if current < UPPER_BOUND:
                    return True
                else:
                    return False
        else:
            return False

    def create_left_child(self, variable, value):
        global UPPER_BOUND
        self.left_child = BnBNode(copy.deepcopy(self.mdl), UPPER_BOUND)
        self.left_child.mdl.add_constraint(
            self.left_child.mdl.optim_planes[int(variable[1])][int(variable[2])] <= int(value))

    def create_right_child(self, variable, value):
        global UPPER_BOUND
        self.right_child = BnBNode(copy.deepcopy(self.mdl), UPPER_BOUND)
        self.right_child.mdl.add_constraint(
            self.right_child.mdl.optim_planes[int(variable[1])][int(variable[2])] >= (int(value)+1))

    def check_is_integer(self):
        solution = self.mdl_solution.as_dict()
        has_float = False
        for key, value in solution.items():
            value = round(value, 5)
            if (not value.is_integer()) and (key == "optim_planes"):
                has_float = True
            if has_float:
                return False
        return True

    def get_float_vars(self):
        solution = self.mdl_solution.as_dict()
        branch_keys = list()
        branch_vars = list()
        for key, value in solution.items():
            value = round(value, 5)
            if (not value.is_integer()) and (key == "optim_planes"):
                branch_keys.append(key)
                branch_vars.append(value)
        return branch_keys, branch_vars

    def get_initial_bound(self):
        global UPPER_BOUND
        initial_mdl = copy.deepcopy(self.mdl)
        solution = initial_mdl.solve(log_output=False)
        print("INITIAL SOLUTION VARS")
        print(dir(initial_mdl))
        model_vars = list(mdl.iter_continuous_vars())

        print(model_vars)
        y_values = list(solution.get_all_values())
        for i in range(len(y_values)):
            initial_mdl.add_constraint(
                initial_mdl.model_vars[i] == int(y_values[i]))
        solution = initial_mdl.solve(log_output=False)
        UPPER_BOUND = solution.objective_value
        return solution.objective_value

    def choose_float_to_branch(self):
        branch_keys, branch_vars = self.get_float_vars()
        frac, _ = np.modf(branch_vars)
        min_idx = frac.argmin()
        max_idx = frac.argmax()
        to_branch = np.array([frac[min_idx], 1 - frac[max_idx]]).argmin()
        if to_branch:
            return branch_keys[max_idx], branch_vars[max_idx], to_branch
        else:
            return branch_keys[min_idx], branch_vars[min_idx], to_branch


def bnb(node):
    print("### Start BnB ###")
    global UPPER_BOUND
    found_solution = node.solve()
    print(UPPER_BOUND)
    if found_solution is True:
        key, value, side = node.choose_float_to_branch()
        # create right child
        if side:
            print("right_branch ", key, value)
            node.create_right_child(key, value)
            answ_right = bnb(node.right_child)
            print("answer ", answ_right)
            print("left_branch ", key, value)
            node.create_left_child(key, value)
            answ_left = bnb(node.left_child)
            print("answer ", answ_left)
        else:
            print("left_branch ", key, value)
            node.create_left_child(key, value)
            answ_left = bnb(node.left_child)
            print("answer ", answ_left)
            print("right_branch ", key, value)
            node.create_right_child(key, value)
            answ_right = bnb(node.right_child)
            print("answer ", answ_right)

        if answ_left is False and answ_right is False:
            return False
        if answ_left is False:
            return answ_right
        if answ_right is False:
            return answ_left
        if answ_left.objective_value < answ_right.objective_value:
            return answ_left
        return answ_right
    elif found_solution is not False:
        return found_solution
    return False


def benchmark(tasks):
    global UPPER_BOUND
    time_per_run = list()
    for _ in range(tasks):
        print("### Start Benchmarking")
        UPPER_BOUND = 0
        mdl.print_information()

        mdl = add_constraints(mdl)
        begin_time = time.time()
        tree = BnBNode(mdl, hot_start=False)
        result = bnb(tree)
        print(result)
        end_time = time.time()
        print(end_time-begin_time)
        time_per_run.append(end_time-begin_time)
    print(time_per_run)
    print("Average time over {} runs:".format(tasks))
    print(sum(time_per_run)/len(time_per_run))


if __name__ == '__main__':

    benchmark(100)
