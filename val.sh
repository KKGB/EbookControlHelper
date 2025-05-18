yolo task=segment mode=val \
     model=runs/segment/train/weights/best.pt \
     data=eye_seg_dataset.yaml \
     batch=16 imgsz=640 device=0 \
     save_txt=True save_json=True