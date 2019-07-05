# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import math

# --- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)


def func1(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total

# --- MAIN ---------------------------------------------------------------------+


class Particle:
    def __init__(self, x0):
        self.position_i = []          # particle position
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.fitness_best_i = -1      # best fitness individual
        self.fitness_i = -1           # fitness individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.fitness_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.fitness_i < self.fitness_best_i or self.fitness_best_i == -1:
            self.pos_best_i = self.position_i
            self.fitness_best_i = self.fitness_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        # constant inertia weight (how much to weigh the previous velocity)
        w = 0.5
        c1 = 1        # cognitive constant
        c2 = 2        # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2*r2*(pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if necessary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


class PSO():
    def __init__(self, costFunc, bounds, num_particles, maxiter):
        global num_dimensions
        num_dimensions = len(bounds)

        self.costFunc = costFunc
        self.bounds = bounds
        self.num_particles = num_particles
        self.maxiter = maxiter
    
    def run(self):
        
        fitness_best_g = -1               # best fitness for group
        pos_best_g = []                   # best position for group

        # establish the swarm
        swarm = []
        for i in range(0, self.num_particles):
            # posição inicial aleatória
            initial = [random.uniform(a, b) for (a, b) in self.bounds]        
            swarm.append(Particle(initial))

        # begin optimization loop
        for i in range(self.maxiter):
    
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, self.num_particles):
                swarm[j].evaluate(self.costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].fitness_i < fitness_best_g or fitness_best_g == -1:
                    pos_best_g = list(swarm[j].position_i)
                    fitness_best_g = float(swarm[j].fitness_i)

            # cycle through swarm and update velocities and position
            for j in range(0, self.num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(self.bounds)

        return fitness_best_g

if __name__ == "__PSO__":
    main()

# --- RUN ----------------------------------------------------------------------+

# input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
#bounds = [(-10, 10), (-10, 10), (-10, 10), (-1, 1)]
#pso = PSO(func1, bounds, num_particles=15, maxiter=30)

#pso.run()

# --- END ----------------------------------------------------------------------+


