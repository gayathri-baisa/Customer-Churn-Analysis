import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Paths:
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_dir: str = os.path.join(project_root, "data")
    raw_dir: str = os.path.join(data_dir, "raw")
    outputs_dir: str = os.path.join(data_dir, "outputs")
    models_dir: str = os.path.join(project_root, "models")
    sql_dir: str = os.path.join(project_root, "sql")


def ensure_dirs_exist() -> None:
    for d in [Paths.data_dir, Paths.raw_dir, Paths.outputs_dir, Paths.models_dir, Paths.sql_dir]:
        os.makedirs(d, exist_ok=True)


DEFAULT_DB_PATH = os.path.join(Paths.data_dir, "churn.db")

