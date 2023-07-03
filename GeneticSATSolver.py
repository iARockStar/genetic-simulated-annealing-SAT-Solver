import random

from matplotlib import pyplot as plt
from pysat.formula import CNF
from pysat.solvers import Solver

formula = CNF(from_file="Input.cnf")

POPULATION_SIZE = 8000
MUTATION_RATE = 0.1
MATE_RATE = 0.95
NUM_GENERATIONS = 100


class Gene(object):

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

    def create_child(self, par2):

        child_chromosome = []
        i = 0
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            parent_chooser = random.random()

            child_chromosome.append(gp1 if parent_chooser < 1 / 3
                                    else gp2)

            i += 1

        return Gene(child_chromosome)

    def cal_fitness(self):

        fitness = 0
        for gs in formula.clauses:
            for lits in gs:
                if lits in self.chromosome:
                    fitness += 1
                    break
        return fitness

    def mutated_genes(self, chromosome):
        mutated_idx = random.randint(0, formula.nv - 1)
        chromosome[mutated_idx] = -chromosome[mutated_idx]
        return chromosome


def main():
    global POPULATION_SIZE
    global MUTATION_RATE
    generation = 1
    population = []

    for _ in range(POPULATION_SIZE):
        population.append(Gene(None))

    scores = []
    avg = []
    for _ in range(NUM_GENERATIONS):
        population = sorted(population, key=lambda x: -x.fitness)

        if population[0].fitness == len(formula.clauses):
            break

        new_generation = []

        new_generation.extend(population[:1])

        for _ in range(7999):
            parent1 = random.choice(population[:100])
            parent2 = random.choice(population[:100])
            if random.random() < MATE_RATE:
                child = parent1.create_child(parent2)
            else:
                child = (parent1 if random.random() < 0.5 else parent2)
            if random.random() < MUTATION_RATE:
                child = (Gene(child.mutated_genes(child.chromosome))
                             if random.random() < MUTATION_RATE
                             else child)
            new_generation.append(child)
        MUTATION_RATE -= 0.02

        population = new_generation
        for gene in population:
            scores.append(gene.fitness)

        print("Generation: " + str(generation) +
              "  Fitness: " + str(population[0].fitness))
        avg.append(sum(scores) / len(scores))
        scores = []
        generation += 1
    if population[0].fitness < len(formula.clauses):
        print(False)
        return

    print("Generation: " + str(generation)
          + "  Fitness: " + str(population[0].fitness))

    solver = Solver()
    solver.append_formula(formula.clauses)
    print(solver.solve(assumptions=population[0].chromosome))
    print(population[0].chromosome)

    plt.plot([i + 1 for i in range(generation - 1)], avg, marker='o')
    plt.xticks(list(range(1, generation)))
    plt.xlabel('gen_num')
    plt.ylabel('avg_score')
    plt.title('Graph showing the increase of scores')
    plt.show()


if __name__ == '__main__':
    main()
