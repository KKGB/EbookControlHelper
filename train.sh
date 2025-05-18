yolo task=segment mode=train \
     model=pretrain/yolo11s-seg.pt \
     data=ultralytics/ultralytics/cfg/datasets/custom.yaml \
     epochs=50 imgsz=640 batch=256 workers=16 save_period=1 weight_decay=0.0001 optimizer=AdamW lr0=0.001 \
     device=0,1,2,3

yolo task=segment mode=val \
     model=ultralytics/runs/segment/train/weights/best.pt \
     data=eye_seg_dataset.yaml \
     batch=256 imgsz=640 device=0,1,2,3 \
     save_txt=True save_json=True

yolo task=segment mode=predict \
     model=ultralytics/runs/segment/train/weights/best.pt \
     source=$(find data/images/val -name '*.jpg' | shuf -n 10) \
     save=True save_masks=True
