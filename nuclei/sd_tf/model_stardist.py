import cv2
import time
import numpy as np
from .utils import export_detections_to_table, map_coords
import numpy as np
import os
from stardist.models import StarDist2D

class StardistSegmentation_tf:
    def __init__(self, configs, device='cpu'):
        self.configs = configs
        self.model = self.load_model(configs.model_path, device=device)
        self.input_size = (self.configs.default_input_size, self.configs.default_input_size)

    def load_model(self, model, device='cpu'):
        stardist_model = StarDist2D(None, name=os.path.basename(model), basedir=os.path.dirname(model))
        return stardist_model

    def get_info(self):
        return {
            'input_name': self.input_name, 
            'output_names': self.output_names, 
            'input_size': self.input_size, 
            'model_dtype': self.model_dtype,
       }

    def __call__(self, images, preprocess=True):
        s0 = time.time()
        if preprocess:
            inputs, image_sizes = self.preprocess(images)
        else:
            assert images.shape[1:] == (3,) + self.input_size, f"Inputs require preprocess."
            inputs, image_sizes = images, None
        s1 = time.time()
        preds = self.predict(inputs)
        s2 = time.time()
        results = self.postprocess(
            preds, image_sizes
        )
        s3 = time.time()
        print(f"preprocess: {s1-s0}, inference: {s2-s1}, postprocess: {s3-s2}")

        return results

    def preprocess(self, images):
        h, w = self.input_size
        inputs, image_sizes = [], []
        for img in images:
            img = np.array(img)
            h_ori, w_ori = img.shape[0], img.shape[1]
            image_sizes.append([h_ori, w_ori])
            if h != h_ori or w != w_ori:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            if img.dtype == np.uint8:
                img = img / 255.0
            inputs.append(img)
        return np.stack(inputs), np.array(image_sizes)

    def predict(self, inputs):
        results = []
        for i in range(len(inputs)):
            label, res = self.model.predict_instances(inputs[i], n_tiles=self.model._guess_n_tiles(inputs[i]))
            results.append(res)

        return results

    def postprocess(self, preds, image_sizes=None):    
        h, w = self.input_size
        res = []  
        for pi in range(len(preds)):
            scores = preds[pi]['prob']
            labels = preds[pi]['class_id']
            boxes = np.zeros((preds[pi]['prob'].shape[0], 4))
            masks = []
            for ci in range(preds[pi]['coord'].shape[0]):
                cur_mask = preds[pi]['coord'][ci].T
                cur_mask[:,[0,1]] = cur_mask[:,[1,0]]
                cur_mask[:,0] *= image_sizes[0,1]/w
                cur_mask[:,1] *= image_sizes[0,1]/w
                masks.append(cur_mask)
                boxes[ci][0] = np.min(cur_mask[:,0])
                boxes[ci][1] = np.min(cur_mask[:,1])
                boxes[ci][2] = np.max(cur_mask[:,0])
                boxes[ci][3] = np.max(cur_mask[:,1])

            o = {'boxes': boxes, 'labels': labels, 'scores': scores, 'masks': masks}    
            res.append(o)

        return res

    def convert_results_to_annotations(self, output, patch_info, annotator=None, extra={}):
        #pdb.set_trace()
        output = map_coords(output, patch_info)

        ## save tables
        st = time.time()
        df = export_detections_to_table(
            output, labels_text=self.configs.labels_text,
            save_masks=True,
        )
        df['xc'] = (df['x0'] + df['x1']) / 2
        df['yc'] = (df['y0'] + df['y1']) / 2
        # df['box_area'] = (df['x1'] - df['x0']) * (df['y1'] - df['y0'])
        df['description'] = df['label'].astype(str) + '; \nscore=' + df['score'].astype(str)
        df = df.drop(columns=['score'])
        df['annotator'] = annotator or self.configs.model_name
        for k, v in extra.items():
            df[k] = v
        print(f"Export table: {time.time() - st}s.")

        return df.to_dict(orient='records')

    def test_run(self, image="http://images.cocodataset.org/val2017/000000039769.jpg"):
        import requests
        from PIL import Image

        print(f"Testing service for {image}")
        st = time.time()
        image = Image.open(requests.get(image, stream=True).raw)
        r = self.__call__([np.array(image)])

        print(f"Test service: {len(r)} ({time.time()-st}s)")
