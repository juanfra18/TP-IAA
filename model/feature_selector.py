from torch import optim
import torch.nn as nn

import numpy as np
import pygad

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.trainer import Trainer
from model.model import Model
from model.loader import CustomDataset
from analysis.data_split import stratified_split
import pandas as pd
from torch.utils.data import DataLoader
from analysis.logger import Logger
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def feature_selector(model : Model, trainer : Trainer, data_header : list[str], 
                     mask_size_weight : float = 0.5, num_generations : int = 25, 
                     population_size : int = 5) -> None:
    
    
    fitness_function = lambda ga_instance,solution,index : -(trainer.train_model_from_scratch(model, solution, False, False,False,False)[1] + mask_size_weight * np.sum(solution) / len(data_header))
    
    num_parents_mating = max(2, population_size // 2)  # Default to half of population
    num_genes = len(data_header)

    pbar = tqdm(total=num_generations, desc="Genetic Algorithm", unit="gen")
    
    def on_generation(ga_instance):
        generation = ga_instance.generations_completed
        best_fitness = ga_instance.best_solutions_fitness[-1] if len(ga_instance.best_solutions_fitness) > 0 else 0
        pbar.update(1)
        pbar.set_postfix({"Best Fitness": f"{best_fitness:.4f}"})
    
    ga_instance = pygad.GA(
        sol_per_pop=population_size,
        num_genes=num_genes,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        gene_space=np.array([0, 1], dtype=int),
        gene_type=int,
        on_generation=on_generation
    )

    ga_instance.run()
    pbar.close()

    fitness_over_time = ga_instance.best_solutions_fitness
    plt.figure()
    plt.plot(fitness_over_time, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Best Solution Fitness")
    plt.title("Best Solution Fitness Over Generations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/feature_selector_fitness_over_time.png")
    plt.close()
    
    solution, solution_fitness, solution_generation = ga_instance.best_solution()
    
    with open("results/feature_selector_solution.txt", "w") as f:
        f.write(str(solution))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Selection using Genetic Algorithm")
    
    parser.add_argument("--BATCH_SIZE", type=int, default=10,
                        help="Batch size for training (default: 10)")
    parser.add_argument("--HIDDEN_SIZES", type=str, default="32,16,1",
                        help="Comma-separated list of hidden layer sizes (default: '32,16,1')")
    parser.add_argument("--DROPOUT_PROBABILITY", type=float, default=0.5,
                        help="Dropout probability (default: 0.5)")
    parser.add_argument("--POPULATION_SIZE", type=int, default=5,
                        help="Population size for genetic algorithm (default: 5)")
    parser.add_argument("--MASK_SIZE_WEIGHT", type=float, default=0.2,
                        help="Weight for mask size in fitness function (default: 0.2)")
    parser.add_argument("--GENERATIONS", type=int, default=25,
                        help="Number of generations for genetic algorithm (default: 25)")
    
    args = parser.parse_args()
    
    # Parse hidden sizes
    HIDDEN_SIZES = [int(x.strip()) for x in args.HIDDEN_SIZES.split(",")]
    
    normalize_output = True
    BATCH_SIZE = args.BATCH_SIZE
    df = pd.read_csv("datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv", encoding="latin1")
    train_df, val_df, test_df = stratified_split(df)

    train_data = CustomDataset(train_df, normalize_output)
    val_data = CustomDataset(val_df, normalize_output)
    test_data = CustomDataset(test_df, normalize_output)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    input_size = train_data.column_count
    
    # Adjust input size in hidden_sizes if needed
    if HIDDEN_SIZES[0] != input_size:
        HIDDEN_SIZES = [input_size] + HIDDEN_SIZES
    
    model = Model(
        weight_path=None,
        description="test",
        input_size=input_size,
        hidden_sizes=HIDDEN_SIZES,
        hidden_activation=nn.ReLU,
        output_activation=nn.Sigmoid,
        dropout_p=args.DROPOUT_PROBABILITY
    )
    
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.MSELoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        num_epochs=25,
        logger=Logger("results/logs")
    )
    
    feature_selector(
        model, 
        trainer, 
        train_data.get_column_names(), 
        mask_size_weight=args.MASK_SIZE_WEIGHT,
        num_generations=args.GENERATIONS,
        population_size=args.POPULATION_SIZE
    )