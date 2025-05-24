# yolo task=segment mode=train \
#      model=pretrain/yolo11m-seg.pt \
#      data=ultralytics/ultralytics/cfg/datasets/custom2.yaml \
#      epochs=50 imgsz=640 batch=128 workers=16 save_period=1 weight_decay=0.0001 optimizer=AdamW lr0=0.001 \
#      device=4,5,6,7 \
#      project=runs/segment/yolo11m

# yolo task=segment mode=val \
#      model=ultralytics/runs/segment/train5/weights/best.pt \
#      data=eye_seg_dataset.yaml \
#      batch=512 imgsz=640 device=0,1,2,3 \
#      save_txt=True save_json=True

# yolo task=segment mode=predict \
#      model=runs/segment/yolo11s/train/weights/best.pt \
#      source=data/images/train/G1/173/30/RGB/NIA_EYE_G1_173_30_RGB_F_0751.jpg \
#      save=True box=False \
#      project=runs/segment/yolo11s


yolo task=segment mode=train \
     model=pretrain/yolo11s-seg.pt \
     data=ultralytics/ultralytics/cfg/datasets/custom2.yaml \
     epochs=100 imgsz=640 batch=128 workers=16 \
     device=4,5,6,7 \
     optimizer=AdamW \
     lr0=0.001 \
     weight_decay=0.0001 \
     box=3 mosaic=0.3 close_mosaic=20 cos_lr=True fliplr=0 \
     save_period=1 \
     project=runs/segment/yolo11s_2