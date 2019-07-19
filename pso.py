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
import numpy as np
from operator import itemgetter

# --- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)


def func1(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total

# --- MAIN ---------------------------------------------------------------------+


class Particle:
    def __init__(self, x0, inertia, constriction):
        self.position_i = []          # particle position
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.fitness_best_i = -1      # best fitness individual
        self.fitness_i = -1           # fitness individual
        self.neighbors = []           # list of other particles ordered by proximity
        self.pos_best_l = []          # best position locally
        self.inertia = inertia        # particle inertia value
        self.constriction = constriction  # 1 if using constriction factor

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
    def update_velocity(self, pos_best_g, num_neighbors):
        # constant inertia weight (how much to weigh the previous velocity)
        w = self.inertia
        c1 = 2.1        # cognitive constant
        c2 = 2.1        # social constant
        phi = c1+c2
        k = 2/(np.absolute(2-phi-np.sqrt(phi**2 - 4*phi)))

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()
            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            if num_neighbors >= 0:
                vel_social = c2 * r2 * \
                    (self.pos_best_l[i] - self.position_i[i])
            else:
                vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            if self.constriction:
                self.velocity_i[i] = k * \
                    (self.velocity_i[i] + vel_cognitive + vel_social)
            else:
                self.velocity_i[i] = w * self.velocity_i[i] + \
                    vel_cognitive + vel_social

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

    # calculate euclidian distance between 2 particles
    def euclidian_distance(self, other_particle):
        coord_self = np.array(self.position_i)
        coord_other = np.array(other_particle.position_i)

        distance = np.linalg.norm(coord_self - coord_other)
        return distance

    # find best position locally, using neighbors (local topology only)
    def find_best_local(self, num_neighbors):
        fitness_best_l = self.fitness_i
        self.pos_best_l = self.position_i
        for i in range(0, num_neighbors):
            if self.neighbors[i]['particle'].fitness_i < fitness_best_l:
                self.pos_best_l = self.neighbors[i]['particle'].position_i


class PSO():
    def __init__(self, costFunc, bounds, num_particles=50, maxiter=100, num_neighbors=-1, inertia=0.5, constriction=False):
        global num_dimensions
        num_dimensions = len(bounds)

        self.costFunc = costFunc
        self.bounds = bounds
        self.num_particles = num_particles
        self.maxiter = maxiter
        self.num_neighbors = num_neighbors
        self.inertia = inertia
        self.constriction = constriction

    def run(self):

        fitness_best_g = -1               # best fitness for group
        pos_best_g = []                   # best position for group
        iter_best_fitness = []            # array of best fitness of each iteration

        # establish the swarm
        swarm = []
        for i in range(0, self.num_particles):
            # posição inicial aleatória
            initial = [random.uniform(a, b) for (a, b) in self.bounds]
            swarm.append(Particle(initial, self.inertia, self.constriction))

        # begin optimization loop
        for i in range(self.maxiter):

            # cycle through particles in swarm and evaluate fitness
            for j in range(0, self.num_particles):
                swarm[j].evaluate(self.costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].fitness_i < fitness_best_g or fitness_best_g == -1:
                    pos_best_g = list(swarm[j].position_i)
                    fitness_best_g = float(swarm[j].fitness_i)

                # find ordered list of neighbors (by distance from closest to farthest
                if self.num_neighbors >= 0:
                    for k in range(0, self.num_particles):
                        if swarm[j] is not swarm[k]:
                            distance = swarm[j].euclidian_distance(swarm[k])
                            swarm[j].neighbors.append(
                                {'particle': swarm[k], 'distance': distance})

                    swarm[j].neighbors.sort(key=itemgetter('distance'))
                    swarm[j].find_best_local(self.num_neighbors)

            # save best fitnesses by iteration
            iter_best_fitness.append(fitness_best_g)

            # cycle through swarm and update velocities and position
            for j in range(0, self.num_particles):
                swarm[j].update_velocity(pos_best_g, self.num_neighbors)
                swarm[j].update_position(self.bounds)

        return fitness_best_g, iter_best_fitness


if __name__ == "__PSO__":
    main()

# --- RUN ----------------------------------------------------------------------+

# input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
#bounds = [(-10, 10), (-10, 10), (-10, 10), (-1, 1)]
#pso = PSO(func1, bounds, num_particles=15, maxiter=30)

# pso.run()

# --- END ----------------------------------------------------------------------+
