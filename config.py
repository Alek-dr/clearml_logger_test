from pathlib import Path
from typing import Union, Any, Tuple, List

from pydantic_yaml import YamlModel


class BaseConfig(YamlModel):
    @classmethod
    def parse_raw(cls, filename: Union[str, Path], *args, **kwargs):
        with open(filename, "r") as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)


class TrainingConfig(BaseConfig):
    # Clearml task arguments
    clearml_project: str
    clearml_task_name: str

    # Clearml task arguments
    data: str
    epochs: int
    batch_size: int