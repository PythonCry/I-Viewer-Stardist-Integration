from .generator import StardistGenerator_tf
from .model_stardist import StardistSegmentation_tf
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any


class StardistConfig_tf(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())

    model_path: str
    server: str = 'stardist_tf'
    device: str = 'cpu'
    batch_size: int = 1
    default_input_size: int = 320
    dzi_settings: Dict[str, Any] = {
        'format': 'jpeg', 
        'tile_size': 512, 
        'overlap': 64, 
        'limit_bounds': False, 
        'tile_quality': 50,
    }
    labels: List[str] = [
        'bg', 'Neutrophil', 'Epithelial', 'Lymphocyte', 
        'Plasma', 'Eosinophil', 'Connective',
    ]
    labels_color: Dict[str, str] = {
        0: "#ffffff", 
        1: "#00ff00", 
        2: "#ff0000", 
        3: "#0000ff", 
        4: "#ff00ff", 
        5: "#ffff00",
        6: "#0094e1",
        #7: "#646464",
    }
    labels_text: Dict[int, str] = {
        0: 'bg', 1: 'Neutrophil', 2: 'Epithelial', 3: 'Lymphocyte', 
        4: 'Plasma', 5: 'Eosinophil', 6: 'Connective',
    }

__all__ = [
    'StardistGenerator_tf', 
    'StardistSegmentation_tf',
    'StardistConfig_tf', 
]
