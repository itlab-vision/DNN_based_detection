import cv2
from operator import itemgetter
import argparse


class ArgsFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(formatter_class=ArgsFormatter,
                                 description="")
parser.add_argument("--result", required=True, help="")
parser.add_argument("--gt", required=True, help="")
parser.add_argument("--group_threshold", type=int, default=0, help="")
parser.add_argument("--group", action="store_true", help="")
args = parser.parse_args()

predictions = []
with file(args.result, "r") as f:
    image_name = f.readline().strip()
    while image_name != "":
        item = {"name": image_name, "bboxes": [], "scores": []}
        detections_num = int(f.readline().strip())
        for _ in range(detections_num):
            detection = map(float, f.readline().strip().split(" "))
            item["bboxes"].append(detection[:4])
            item["scores"].append(detection[4])
        predictions.append(item)
        image_name = f.readline().strip()

print(len(predictions))

gt = [{"name": str(i), "bboxes": []} for i in range(len(predictions))]
with file(args.gt, "r") as f:
    s = f.readline().strip()
    while s != "":
        values = s.split(" ")
        # assert values[1] == "CAR"
        idx = int(values[0])
        bbox = map(float, values[2:])
        bbox[2] -= bbox[0]
        bbox[3] -= bbox[1]
        gt[idx]["bboxes"].append(bbox)
        s = f.readline().strip()

print(len(gt))

def intersection_area(b1, b2):
    # print(b1, b2)
    x = max(b1[0], b2[0])
    y = max(b1[1], b2[1])
    w = min(b1[2] + b1[0], b2[2] + b2[0]) - x
    h = min(b1[3] + b1[1], b2[3] + b2[1]) - y
    # b1_cap_b2 = [x, y, w, h]
    return w * h


def match_bbox(bbox, gts, threshold=0.5):
    for i, gt_bbox in enumerate(gts):
        intersection = intersection_area(bbox, gt_bbox)
        union = bbox[2] * bbox[3] + gt_bbox[2] * gt_bbox[3] - intersection
        if intersection / union > threshold:
            del gts[i]
            return True
    return False


if args.group:
    for pred in predictions:
        pred["bboxes"], pred["scores"] = cv2.groupRectangles(pred["bboxes"], args.group_threshold, eps=0.2)

total_preds = 0
total_gt = 0
fp = 0
tp = 0
assert len(predictions) == len(gt)
for bbox_pred, bbox_gt in zip(predictions, gt):
    total_preds += len(bbox_pred["bboxes"])
    total_gt += len(bbox_gt["bboxes"])
    if len(bbox_pred["bboxes"]) > 0:
        bbox_pred["bboxes"] = zip(*sorted(zip(bbox_pred["bboxes"],
                                              bbox_pred["scores"]),
                                          key=itemgetter(1),
                                          reverse=True))[0]
    for bbox in bbox_pred["bboxes"]:
        if match_bbox(bbox, bbox_gt["bboxes"]):
            tp += 1
        else:
            fp += 1

print("TPR = {:.2%}".format(float(tp) / total_gt))
print("FDR = {:.2%}".format(float(fp) / total_preds))
print("FPF = {:.2%}".format(float(fp) / len(predictions)))

# TPR (true positive rate) = detected/total_number
# FDR (false detection rate) = false_positives/(detected+false_positives)
# FPF (false positive per frame) = false_positives/frames_number.
