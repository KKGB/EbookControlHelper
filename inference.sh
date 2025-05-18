yolo task=segment mode=predict \
     model=runs/segment/train/weights/best.pt \
     source=$(find data/images/val -name '*.jpg' | shuf -n 20) \
     save=True save_masks=True
