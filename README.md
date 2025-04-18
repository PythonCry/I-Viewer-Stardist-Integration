

# Integrating StarDist into I-Viewer for Cell Segmentation and Classification

This repository demonstrates how to **integrate a custom cell segmentation/classification model** into [I-Viewer](https://github.com/impromptuRong/iviewer_copilot). As an example, it shows how to plug in a **StarDist model pretrained on the [CoNIC Challenge](https://conic-challenge.grand-challenge.org/) dataset** for **nuclei segmentation** and **classification** within H&E images.

![Alt Text](images/I-Viewer-Stardist.png)

---

## Overview

- Uses `StarDist2D` from the [`stardist`](https://github.com/stardist/stardist) Python package
- Model trained on CoNIC: [Colon Nuclei Identification and Counting](https://conic-challenge.grand-challenge.org/)

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
  + Purpose: Resizes each image tile to the model’s expected input size and normalizes the pixel values.
  + Input: List of raw RGB image tiles.
  + Output: List of normalized images as the model’s inputs.

- **Prediction**:  
  Uses `model.predict_instances()` to obtain the predictions using Stardist.

- **Postprocessing**:  
  + Purpose: Converts model output to I-Viewer's expected format with:  
  + Input:   
    + `preds`: Predictions from Stardist model.  
    + `image_sizes`: numpy array of shape `(B, 2)` with original image tile sizes (height, width) for rescaling polygon coordinates back to original scale.  
  + Output: A list of prediction results -- one for each input image tile: `[pred0, pred1, pred2, ...]`.  
  Each predX is a Python dictionary with keys in the following table, and Nc is the count of the detected cells in that tile. 

| Key     | Type        | Shape/Structure      |Description                        |
|---------|-------------|------------------|------------------|
| `boxes` | `np.ndarray` | (Nc, 4)         |bounding boxes per cell: `[xmin, ymin, xmax, ymax]`    |
| `labels` | `np.ndarray` | (Nc,)        |Class index for each instance       |
| `scores` | `np.ndarray` | (Nc,)        |Confidence score per detection      |
| `masks`  | `List[np.ndarray]` | Each element (Nc,2) |Polygon coordinates for each cell |

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
2. Update `docker-compose.yml` to Add Stardist Service

---

## Quick Start: Testing the Stardist Integration
 
Follow these steps to verify that the Stardist segmentation layer is working correctly:
 
1. **Open the test page**  
   In your web browser, open `template/index.html`.
 
2. **Select the Stardist model**  
   - Click the model dropdown (the ▼ next to the current model name).  
   - Choose **Stardist**.  
   - Click the **Run** button (▶️) to launch Stardist.
 
3. **Navigate the slide**  
   Pan and zoom by dragging and scrolling over the image.
 
4. **Load the stardist‑tf layer**  
   - Click the **last gray (no‑icon) floating button** at the top‑left of the viewer to open the Annotators panel.  
   - In the panel, select **stardist‑tf**.  
   - Click **OK** to apply.
 
5. **View the results**  
   The **stardist‑tf** segmentation overlay should now appear on the slide.

---

## License

The software package is [licensed](https://github.com/impromptuRong/iviewer_copilot/blob/master/LICENSE). 
For commercial use, please contact [Ruichen Rong](Ruichen.Rong@UTSouthwestern.edu), [Xiaowei Zhan](mailto:Xiaowei.Zhan@UTSouthwestern.edu) and
[Guanghua Xiao](mailto:Guanghua.Xiao@UTSouthwestern.edu).

