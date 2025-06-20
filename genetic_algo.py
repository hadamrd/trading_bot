import random
import logging
from multiprocessing import Pool

import numpy as np
from deap import base, creator, tools, algorithms

from tradingbot2.TradingStrategyEvaluator import TradingStrategyEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneticAlgorithm:
    def __init__(self, config, evaluator: TradingStrategyEvaluator):
        self.config = config
        self.evaluator = evaluator
        self.setup_toolbox()
        self.fitness_cache = {}

    def setup_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        for key, value in self.config['individuals_genes'].items():
            if value['type'] == 'integer':
                self.toolbox.register(f"attr_{key}", random.randint, value['range'][0], value['range'][1])
            elif value['type'] == 'float':
                self.toolbox.register(f"attr_{key}", random.uniform, value['range'][0], value['range'][1])
            elif value['type'] == 'boolean':
                self.toolbox.register(f"attr_{key}", random.randint, 0, 1)

        self.toolbox.register(
            "individual", 
            tools.initCycle, 
            creator.Individual,
            [getattr(self.toolbox, f"attr_{key}") for key in self.config['individuals_genes'].keys()], 
            n=1
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.custom_mutate, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=self.config['population_config']['tournament_size'])
        self.toolbox.register("evaluate", self.evaluator.evaluate)

    def custom_mutate(self, individual, indpb):
        for i, (key, value) in enumerate(self.config['individuals_genes'].items()):
            if random.random() < indpb:
                if value['type'] == 'integer':
                    individual[i] = random.randint(value['range'][0], value['range'][1])
                elif value['type'] == 'float':
                    individual[i] = random.uniform(value['range'][0], value['range'][1])
                elif value['type'] == 'boolean':
                    individual[i] = random.randint(0, 1)
        return individual,

    def parallel_evaluate(self, individuals):
        with Pool() as pool:
            return pool.map(self.evaluator.evaluate, individuals)

    def run(self):
        population = self.toolbox.population(n=self.config['population_config']['population_size'])

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        # Evaluate the initial population
        fitnesses = self.parallel_evaluate(population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        for gen in range(self.config['population_config']['generations']):
            offspring = algorithms.varAnd(
                population, 
                self.toolbox, 
                cxpb=self.config['population_config']['crossover_prob'], 
                mutpb=self.config['population_config']['mutation_prob']
            )

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.parallel_evaluate(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            population = self.toolbox.select(offspring, k=len(population))

            # Append the current generation statistics to the logbook
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            logger.info(logbook.stream)

        best_ind = tools.selBest(population, k=1)[0]
        best_params = dict(zip(self.config['individuals_genes'].keys(), best_ind))

        return best_params, logbook