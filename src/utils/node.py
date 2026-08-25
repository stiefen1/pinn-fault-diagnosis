from typing import Any, Dict, List
from src.utils.random import sample_clipped_value, sample_fault_cfg

import numpy as np
from copy import deepcopy

def is_distribution_node(node: Dict) -> bool:
    return "min" in node and "max" in node

def is_fault_node(node: Dict) -> bool:
    return "prob" in node and "amplitude" in node and "time" in node

def sample_node(rng: np.random.Generator, node: Any) -> Any:
    if isinstance(node, Dict):
        sampled: dict[str, Any] = {}
        for key, subnode in node.items():
            if is_distribution_node(node):
                val = sample_clipped_value(rng, node) # sample float value according to spec

                # Set type to integer if indicated
                if node.get("type", "float") == "int":
                    if isinstance(val, List):
                        val = [int(v) for v in val]
                    elif isinstance(val, float):
                        val = int(val)

                # clear out keys related to distribution since value has been chosen
                distribution_keys = ['max', 'min', 'mean', 'std', 'type']
                node_copy = deepcopy(node)
                for k in distribution_keys:
                     node_copy.pop(k, None)

                return val
            elif is_fault_node(node):
                return sample_fault_cfg(rng, node)
            else:
                sampled[key] = sample_node(rng, subnode)
        return sampled

    if isinstance(node, List):
        return [sample_node(rng, subnode) for subnode in node]

    return node