import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    print("DONE")


if __name__ == "__main__":
    main()
