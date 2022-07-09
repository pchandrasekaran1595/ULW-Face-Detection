import os
import re
import sys
import cv2
import onnx
import platform
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(BASE_PATH, 'input')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
MODEL_PATH  = os.path.join(BASE_PATH, 'models')

if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)


def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def get_image(path: str) -> np.ndarray:
    return cv2.cvtColor(src=cv2.imread(path, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)

  
def show_image(image, cmap: str = "gnuplot2", title: str=None) -> None:
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    if title: plt.title(title)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()

######################################################################################################################


"""
        https://github.com/onnx/models/blob/master/vision/body_analysis/ultraface/dependencies/box_utils.py 

"""


def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

######################################################################################################################

class CFG(object):
    def __init__(self, size: int=320) -> None:
        self.size = size
        self.mean = [127, 127, 127]
        if self.size == 320:
            self.path = os.path.join(MODEL_PATH, "ULWFD-320.onnx")
        else:
            self.path = os.path.join(MODEL_PATH, "ULWFD-640.onnx")
        ort.set_default_logger_severity(3)
    
    def setup(self) -> None:
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def infer(self, image: np.ndarray) -> tuple:
        h, w, _ = image.shape
        if self.size == 320: image = cv2.resize(src=image, dsize=(320, 240), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        else: image = cv2.resize(src=image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        for i in range(image.shape[0]):
            image[i, :, :] = (image[i, :, :] - self.mean[i]) / 128
        image = np.expand_dims(image, axis=0)
        input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}
        result = self.ort_session.run(None, input)

        boxes, scores = result[1].squeeze(), result[0].squeeze()
        box_scores = np.concatenate((boxes, scores[:, 1].reshape(-1, 1)), axis=1)
        nmsboxes = hard_nms(box_scores, 0.99)

        del boxes, scores, box_scores

        best_score_index = 0
        x1, y1, x2, y2 = nmsboxes[best_score_index][0] * w, \
                         nmsboxes[best_score_index][1] * h, \
                         nmsboxes[best_score_index][2] * w, \
                         nmsboxes[best_score_index][3] * h
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return (x1, y1, x2, y2)


def main():
    args_1: tuple = ("--mode", "-m")
    args_2: tuple = ("--file", "-f")
    args_3: tuple = ("--downscale", "-ds")
    args_4: tuple = ("--save", "-s")

    mode: str = "image"
    filename: str = "Test_1.jpg"
    downscale: float = None
    save: bool = False

    if args_1[0] in sys.argv: mode = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: mode = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: filename = sys.argv[sys.argv.index(args_2[0]) + 1]
    if args_2[1] in sys.argv: filename = sys.argv[sys.argv.index(args_2[1]) + 1]

    if args_3[0] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_3[0]) + 1])
    if args_3[1] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_3[1]) + 1])

    if args_4[0] in sys.argv or args_4[1] in sys.argv: save = True

    cfg = CFG()
    cfg.setup()

    if re.match(r"image", mode, re.IGNORECASE):
        image = get_image(os.path.join(INPUT_PATH, filename))
        x1, y1, x2, y2 = cfg.infer(image)

        if save: 
            cv2.imwrite(os.path.join(OUTPUT_PATH, filename[:-4] + " - Result.png"), cv2.cvtColor(src=image[y1:y2, x1:x2], code=cv2.COLOR_BGR2RGB))
        else: 
            disp_image = image.copy()
            cv2.rectangle(disp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            show_image(image=disp_image, title="Detections")
    
    elif re.match(r"video", mode, re.IGNORECASE):
        cap = cv2.VideoCapture(os.path.join(INPUT_PATH, filename))

        while True:
            ret, frame = cap.read()
            if not ret: break
            if downscale:
                frame = cv2.resize(src=frame, dsize=(int(frame.shape[1]/downscale), int(frame.shape[0]/downscale)), interpolation=cv2.INTER_AREA)
            x1, y1, x2, y2 = cfg.infer(cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)) 
            disp_frame = frame.copy()
            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)        
            cv2.imshow("Detection Feed", disp_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif re.match(r"realtime", mode, re.IGNORECASE):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret: continue
            x1, y1, x2, y2 = cfg.infer(cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)) 
            disp_frame = frame.copy()
            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)        
            cv2.imshow("Detection Feed", disp_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        breaker()
        print("--- Unknown Mode ---".upper())
        breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)
