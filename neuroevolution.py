import random

# Function to perform crossover between two neural architectures
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Function to perform mutation on a neural architecture
def mutate(architecture):
    mutation_point = random.randint(0, len(architecture) - 1)
    architecture[mutation_point] = random.randint(1, 10)  # Example mutation, replace with your own logic
    return architecture

# Function to create a random neural architecture with dynamic layers and units
def create_neural_architecture():
    num_layers = random.randint(1, 5)  # Random number of layers between 1 and 5
    architecture = [random.randint(1, 10) for _ in range(num_layers)]  # Random number of units per layer (1 to 10)
    return architecture

# Function to evaluate the fitness of a neural architecture
def evaluate_fitness(architecture):
    fitness = random.random()  # Example fitness evaluation, replace with your actual evaluation code
    return fitness

# Neuroplasticity: Allow the best-performing model to grow its architecture
def grow_architecture(architecture):
    if random.random() < GROWTH_RATE:
        new_layer_units = random.randint(1, 10)  # Add a new layer with random number of units (1 to 10)
        architecture.append(new_layer_units)

# Neuroplasticity: Allow underperforming models to prune their architecture
def prune_architecture(architecture):
    if random.random() < PRUNING_RATE:
        if len(architecture) > 1:
            layer_to_remove = random.randint(0, len(architecture) - 1)
            architecture.pop(layer_to_remove)

# Genetic Algorithm Parameters
POPULATION_SIZE = 5
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1
GROWTH_RATE = 0.2
PRUNING_RATE = 0.2

# Create the initial population of neural architectures
population = [create_neural_architecture() for _ in range(POPULATION_SIZE)]

# Evolution loop
for generation in range(NUM_GENERATIONS):
    # Evaluate the fitness of each neural architecture
    fitness_scores = [evaluate_fitness(architecture) for architecture in population]

    # Sort the population based on fitness (best to worst)
    sorted_population = [arch for _, arch in sorted(zip(fitness_scores, population), reverse=True)]

    # Neuroplasticity: Apply growth to the best model with a certain probability
    grow_architecture(sorted_population[0])

    # Neuroplasticity: Apply pruning to underperforming models with a certain probability
    for i in range(1, len(sorted_population)):
        prune_architecture(sorted_population[i])

    # Create the next generation using crossover and mutation
    new_generation = sorted_population.copy()
    while len(new_generation) < POPULATION_SIZE:
        parent1 = random.choice(sorted_population)
        parent2 = random.choice(sorted_population)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_generation.extend([child1, child2])

    # Replace the old population with the new generation
    population = new_generation

# Final population after evolution
print("Final population:")
for i, architecture in enumerate(population):
    print(f"Neural Architecture {i + 1}: {architecture}")

# Possibly translate code to PyTorch Geometric, there are advantages to PyTorch in comparison to Tensorflow and Spektral Layers
# It wouldnt work if the architectures have different number of layers 