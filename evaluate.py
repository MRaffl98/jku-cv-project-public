from config import *
from typing import Dict, List
from evaluation.utils import BoundingBox, compute_AP, read_bb


def evaluate(detections: Dict[str, List[BoundingBox]], targets: Dict[str, List[BoundingBox]]) -> float:
    return compute_AP(detections, targets)


if __name__ == '__main__':
    detections = read_bb(bbs_path[subset])
    targets = read_bb(target_path[subset])
    ap = evaluate(detections, targets)
    print(f"Average precision={ap:.5f} on {subset} set.")
