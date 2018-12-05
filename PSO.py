import random
import sys
import numpy as np

# WK = 0.9
# C1 = 10
# C2 = 10

# def func1(x):
#     # total = 0
#     # for i in range(len(x)):
#     #     total += x[i]**2
#     # return total
#     return np.sum(x**2)

class Particle:
    def __init__(self, NUM_DIM, WK, C1, C2):
        self.NUM_DIM = NUM_DIM
        self.WK = WK
        self.C1 = C1
        self.C2 = C2
        self.position = np.random.rand(NUM_DIM, 1)
        self.velocity = np.random.rand(NUM_DIM, 1)
        self.best_position = np.array([])
        self.best_error = sys.maxsize

        # for dim in range(0, NUM_DIM):
        #     self.position.append(random.random())
        #     self.velocity.append(0.01 * random.random())

    def evaluate_function(self, cost_function, params):
        error = -1*cost_function(self.position, *params)
        if error < self.best_error:
            self.best_position = self.position
            self.best_error = error

    def update_velocity(self, best_global_position):
        R1 = np.random.rand(self.NUM_DIM,1)
        R2 = np.random.rand(self.NUM_DIM,1)
        self.velocity = self.WK*self.velocity + self.C1*R1*(self.best_position - self.position) + self.C2*R2*(best_global_position - self.position)

    def update_position(self):
        self.position = self.position + self.velocity

class PSO():
    def __init__(self, num_dim, num_particles, max_iter, WK, C1, C2, cost_function, function_params):
        self.num_dim = num_dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.cost_function = cost_function
        self.params = function_params
        self.WK = WK
        self.C1 = C1
        self.C2 = C2
        self.swarm = []
        self.best_global_error = sys.maxsize
        self.best_global_position = np.array([])
        self.num_particles = num_particles

    def run_pso(self):
        for each_particle in range(0, self.num_particles):
            particle = Particle(self.num_dim, self.WK, self.C1, self.C2)
            self.swarm.append(particle)

        for iteration in range(0, self.max_iter):

            for particle in self.swarm:
                particle.evaluate_function(self.cost_function, self.params)

                # PREGUNTAR SI EL BEST DE ESTA PARTICULA ES LMEJOR QUE EL BEST GLOBALS
                if particle.best_error <  self.best_global_error:
                    best_global_error = particle.best_error
                    best_global_position = particle.position

            for particle in self.swarm:
                particle.update_velocity(best_global_position)
                particle.update_position()

        # print(best_global_position)
        # print(best_global_error)
        return (best_global_position, -1*best_global_error)

# def main():
#     particle = PSO(2, 3, 3, func1, 0.9, 10, 10)
#     values = particle.run_pso()

# main()