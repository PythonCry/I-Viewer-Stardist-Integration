
# Integrating StarDist into I-Viewer for Cell Segmentation and Classification

This project demonstrates how to **integrate a custom cell segmentation/classification model** into [I-Viewer](https://github.com/impromptuRong/iviewer_copilot). As an example, it shows how to plug in a **StarDist model pretrained on the CoNIC Challenge dataset** for **nuclei segmentation** and **classification** within H&E images.

---

## Overview

- Uses `StarDist2D` from the [`stardist`](https://github.com/stardist/stardist) Python package
- Model trained on CoNIC: [Colon Nuclei Identification and Counting](https://conic-challenge.grand-challenge.org/)

---

## Project Structure

```bash
.
├── nuclei/
│   ├── sd_tf/
│   │   ├── __init__.py              # Model config (StardistConfig_tf)
│   │   ├── generator.py             # Data pipeline (StardistGenerator_tf)
│   │   ├── model_stardist.py       # Main logic (StardistSegmentation_tf)
│   └── model_registry.py           # Service registration
├── ckpts/
│   └── nuclei-stardist/
│       └── stardist_conic/         # Pretrained model files
└── README.md
```

---

## Steps to Integrate a New Model

### 1. Update `StardistConfig_tf` in `nuclei/sd_tf/__init__.py`

Define server name, tile size, class labels, and device.

```python
class StardistConfig_tf(BaseModel):
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
    ...
    labels: List[str] = ['bg', 'Neutrophil', 'Epithelial', 'Lymphocyte', 'Plasma', 'Eosinophil', 'Connective']
    labels_color: Dict[str, str] = {0: "#ffffff", 1: "#00ff00", ...}
    labels_text: Dict[int, str] = {0: 'bg', 1: 'Neutrophil', ...}
```

---

### 2. Rename or Extend the Generator Class

The `StardistGenerator_tf` class in `generator.py` prepares image tiles. You can reuse existing logic or rename parameters as needed.

---

### 3. Implement the Inference Pipeline

In `model_stardist.py`, define the `StardistSegmentation_tf` class with the following stages:

- **Initialization**:
 ```python
 class StardistSegmentation_tf:
    def __init__(self, configs, device='cpu'):
        self.configs = configs
        self.model = self.load_model(configs.model_path, device=device)
        self.input_size = (self.configs.default_input_size, self.configs.default_input_size)
 ```

- **Model loading**:  
  ```python
    def load_model(self, model, device='cpu'):
        stardist_model = StarDist2D(None, name=os.path.basename(model), basedir=os.path.dirname(model))
        return stardist_model
  ```

- **Preprocessing**:  
  Purpose: Resizes each image tile to the model’s expected input size and normalizes the pixel values.
  Input: List of raw RGB image tiles.
  Output: List of normalized images as the model’s inputs.

- **Prediction**:  
  Uses `model.predict_instances()` to obtain the predictions using Stardist.

- **Postprocessing**:  
  Purpose: Converts model output to I-Viewer's expected format with:
  Input: 
    `preds`: Predictions from Stardist model.
    `image_sizes`: numpy array of shape `(B, 2)` with original image tile sizes (height, width) for rescaling polygon coordinates back to original scale.
  Output: A list of prediction results -- one for each input image tile: `[pred0, pred1, pred2, ...]`. Each predX is a Python dictionary with keys:

| Key     | Type        | Description                        |
|---------|-------------|------------------------------------|
| `boxes` | `np.ndarray` | (Nc, 4) bounding boxes per cell     |
| `labels` | `np.ndarray` | Class index for each instance       |
| `scores` | `np.ndarray` | Confidence score per detection      |
| `masks`  | `List[np.ndarray]` | Polygon coordinates for each cell |

---

### 4. Register the Model

In `nuclei/model_registry.py`:

```python
from sd_tf import StardistSegmentation_tf, StardistGenerator_tf, StardistConfig_tf

MODEL_REGISTRY.register("stardist_tf", "model", StardistSegmentation_tf)
MODEL_REGISTRY.register("stardist_tf", "generator", StardistGenerator_tf)

AGENT_CONFIGS = {
    'stardist-tf': StardistConfig_tf(
        model_path = "./ckpts/nuclei-stardist/stardist_conic"
    )
}
```

---

## Run the Integration

1. Write a docker file for the new model as `nuclei/Dockerfile.stardist_tf`
2. Update docker-compose.yml to Add Stardist Service

---

