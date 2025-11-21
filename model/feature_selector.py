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

def feature_selector(model : Model, trainer : Trainer, data_header : list[str], mask_size_weight : float = 0.5) -> dict[str, bool]:
    
    
    fitness_function = lambda ga_instance,solution,index : -(trainer.train_model_from_scratch(model, solution, False, False,False,False)[1] + mask_size_weight * np.sum(solution) / len(data_header))
    
    num_generations = 25
    num_parents_mating = 5
    num_genes = len(data_header)

    pbar = tqdm(total=num_generations, desc="Genetic Algorithm", unit="gen")
    
    def on_generation(ga_instance):
        generation = ga_instance.generations_completed
        best_fitness = ga_instance.best_solutions_fitness[-1] if len(ga_instance.best_solutions_fitness) > 0 else 0
        pbar.update(1)
        pbar.set_postfix({"Best Fitness": f"{best_fitness:.4f}"})
    
    ga_instance = pygad.GA(
        sol_per_pop=5,
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
    
    return {header : bool(solution[i]) for i, header in enumerate(data_header)}

if __name__ == "__main__":
    
    normalize_output = True
    BATCH_SIZE = 10
    df = pd.read_csv("datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv", encoding="latin1")
    train_df, val_df, test_df = stratified_split(df)

    train_data = CustomDataset(train_df, normalize_output)
    val_data = CustomDataset(val_df, normalize_output)
    test_data = CustomDataset(test_df, normalize_output)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    
    HIDDEN_SIZES = [32, 16, 1]
    input_size = train_data.column_count
    
    model = Model(
        weight_path=None,
        description="test",
        input_size=input_size,
        hidden_sizes=HIDDEN_SIZES,
        hidden_activation=nn.ReLU,
        output_activation=nn.Sigmoid,
        dropout_p=0.5
    )
    
    df = pd.read_csv("datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv", encoding="latin1")
    train_df, val_df, test_df = stratified_split(df)
    
    train_data = CustomDataset(train_df, normalize_output)
    val_data = CustomDataset(val_df, normalize_output)
    test_data = CustomDataset(test_df, normalize_output)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    
    
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.MSELoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        num_epochs=25,
        logger=Logger("results/logs")
    )
    
    feature_selector(model, trainer, train_data.get_column_names(), 0.2)