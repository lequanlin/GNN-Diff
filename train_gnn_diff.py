import hydra
from omegaconf import DictConfig
from core.runner.runner import *


@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_for_data(config: DictConfig):
    config.output_dir = 'outputs/' + config.task.data.dataset

    if config.mode == 'train':

        # Parameter collection
        result_data = train_task_for_data(config)

        # Train GNN-Diff
        result = train_generation(config)

    elif config.mode == 'test':
        for i in range(10):
            result = test_generation(config)
    return


if __name__ == "__main__":
    training_for_data()
