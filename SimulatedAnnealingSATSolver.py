import random
import math

from matplotlib import pyplot as plt
from pysat.formula import CNF
from pysat.solvers import Solver


class Individual(object):

    def __init__(self, chromosome):
        self.literals = []
        for i in range(formula.nv):
            self.literals.append((i + 1))
        # Generate random list with positive or negative integers
        if chromosome is not None:
            self.chromosome = chromosome
        else:
            self.chromosome = [num if random.choice([True, False])
                               else -num for num in self.literals]

        self.fitness = self.cal_fitness()

    def cal_fitness(self):

        fitness = 0
        for gs in formula.clauses:
            for lits in gs:
                if lits in self.chromosome:
                    fitness += 1
                    break
        return fitness


def simulated_annealing(formula, max_iterations,
                        initial_temperature, cooling_rate):


    current_individual = Individual(None)
    current_cost = current_individual.fitness
    current_solution = current_individual.chromosome


    best_solution = current_solution.copy()
    best_cost = current_cost


    temperature = initial_temperature
    itr = 1

    scores = []
    # Iterate for the specified number of iterations
    for iteration in range(max_iterations):
        # Generate a neighbor solution by flipping a random variable
        neighbor_solution = current_solution.copy()

        index = random.randint(0, formula.nv - 1)
        neighbor_solution[index] = -neighbor_solution[index]

        neighbor_cost = Individual(neighbor_solution).fitness

        # Calculate the cost difference between the current and neighbor solutions
        cost_difference = neighbor_cost - current_cost

        # Determine whether to accept the neighbor solution
        if cost_difference > 0 or \
                random.random() < math.exp(-abs(cost_difference) / temperature):
            current_solution = neighbor_solution
            current_cost = neighbor_cost

        # Update the best solution
        if current_cost > best_cost:
            best_solution = current_solution.copy()
            best_cost = current_cost
            if best_cost == len(formula.clauses):
                print("best cost is " + str(best_cost) + " so found!!!")
                break
            print("best cost is " + str(best_cost))

        # Decrease the temperature
        temperature *= cooling_rate
        itr += 1
        scores.append(best_cost)

    return best_solution, itr, scores


# Read the CNF from a file
formula = CNF(from_file="Input.cnf")

# Solve the CNF using simulated annealing

result = simulated_annealing(formula,
                             max_iterations=50000,
                             initial_temperature=2.0,
                             cooling_rate=0.80)
solution = result[0]
itr = result[1]
scores = result[2]
solver = Solver()
solver.append_formula(formula.clauses)
print(solver.solve(assumptions=solution))
# Print the solution
print(solution)

plt.plot([i + 1 for i in range(itr - 1)], scores)
plt.xlabel('Number Of Iterations')
plt.ylabel("The individual's score")
plt.title('Graph showing the individual finding the Global Maximum')
plt.show()
