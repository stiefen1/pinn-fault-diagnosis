from typing import List, Dict, Optional, Any
from pathlib import Path

from src.utils.configs import load_config
from src.utils.random import set_seed
from src.utils.node import sample_node

import numpy as np, json, os

from src.diagnosis import *

class EnvCfg:
    cfg: Dict
    path_to_cfg: Path

    def __init__(
            self,
            cfg: Dict,
    ):
        self.cfg = cfg
        self.reset(seed=self.cfg.get("seed", 42))

    @staticmethod
    def from_yaml(path_to_cfg: Path | str) -> "EnvCfg":
        if isinstance(path_to_cfg, str):
            path_to_cfg = Path(path_to_cfg)
        return EnvCfg(cfg=load_config(path_to_cfg)["env"])

    def reset(self, seed: Optional[int] = None) -> None:
        set_seed(seed=seed) # type: ignore

        if seed is not None:
            self.base_seed = seed

        # scenario seed is incremented when a new scenario is being generated
        self.scenario_seed = self.base_seed
        
    def sample_scenario(self, seed: int) -> Dict:
        rng = np.random.default_rng(seed)
        scenario_cfg = sample_node(rng, self.cfg)
        scenario_cfg["seed"] = seed
        return scenario_cfg

    def __getitem__(self, item: Any) -> Any:
        return self.cfg[item]

    def __call__(self, n: int = 1, save_dir: Optional[str] = None, filenames: str = 'scenario_') -> List[Dict]:
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        scenarios = []
        for i in range(n):
            self.scenario_seed += 1
            scenario = self.sample_scenario(seed=self.scenario_seed)
            if save_dir is not None:
                with open(os.path.join(save_dir, filenames + str(i) + '.json'), "w") as f:
                    json.dump(scenario, f, indent=4)
            scenarios.append(scenario)
        return scenarios

    def as_dict(self) -> Dict:
        return self.cfg

if __name__ == "__main__":
    import pathlib, pprint

    path_to_cfg = pathlib.Path("configs/eval.yaml")
    cfg = EnvCfg.from_yaml(path_to_cfg=path_to_cfg)
    scenario_cfg = cfg()[0]

    pprint.pprint(scenario_cfg)