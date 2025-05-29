yolo task=segment mode=train \
     model=pretrain/yolo11n-seg.pt \
     data=ultralytics/ultralytics/cfg/datasets/custom2.yaml \
     epochs=50 imgsz=640 batch=512 workers=16 \
     device=4,5,6,7 \
     optimizer=AdamW \
     lr0=0.001 \
     weight_decay=0.0001 \
     box=3 mosaic=0 cos_lr=True fliplr=0 \
     save_period=1 \
     project=runs/segment/yolo11n