import random
import pysat
from pysat.formula import CNF
from pysat.solvers import Solver

# two cnf files, one is satisfiable and the other is unsatisfiable
Uformula = CNF(from_file="UInput.cnf")
formula = CNF(from_file="Input.cnf")

# Create a solver instance for unsatisfiable formula
usolver = Solver()
usolver.append_formula(Uformula.clauses)

# Create a solver instance for satisfiable formula
solver = Solver()
solver.append_formula(formula.clauses)

# actually solves it and can find out if it's satisfiable
print(usolver.solve())
print(solver.solve())

# if it's satisfiable we can get an answer
print(usolver.get_model())
print(solver.get_model())

# we can make a test list and feed it to the model as an assumption
variables = []
# test for satisfiable formula:
for i in range(1, formula.nv+1):
    var = (random.randint(0, 1))
    if var == 1:
        variables.append(i)
    else:
        variables.append(-i)

print(solver.solve(assumptions=variables))
