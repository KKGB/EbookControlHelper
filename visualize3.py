import cv2, random, itertools, glob, os, json
from pathlib import Path
from shapely.geometry import Polygon
import numpy as np

ROOT = 'data'         # 데이터셋 root
IMG_DIR = ROOT/'images/train' # 이미지 폴더
LBL_DIR = ROOT/'labels/train' # txt 폴더
OUT_DIR = Path('viz'); OUT_DIR.mkdir(exist_ok=True)

def poly_orientation(pts):
    # Shoelace sum: >0 => CCW, <0 => CW
    return np.sign(sum((x2-x1)*(y2+y1) for (x1,y1),(x2,y2) in
                       zip(pts, pts[1:]+pts[:1])))

samples = random.sample(list(IMG_DIR.rglob('*.jpg')), 20)
for img_path in samples:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    txt = LBL_DIR / img_path.relative_to(IMG_DIR).with_suffix('.txt')

    if not txt.exists(): continue
    for ln in txt.read_text().splitlines():
        cls,*nums = map(float, ln.split())
        pts = [(nums[i]*w, nums[i+1]*h) for i in range(0,len(nums),2)]
        poly = np.array(pts, np.int32)

        # 시각화
        cv2.polylines(img, [poly], isClosed=True, thickness=2, color=(0,255,0))
        cv2.putText(img, str(int(cls)), poly[0], cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,255), 2, cv2.LINE_AA)

        # 유효성·방향 로그
        pg = Polygon(pts)
        print(f'{txt.name}: valid={pg.is_valid}, ',
              'CCW' if poly_orientation(pts)>0 else 'CW')

    cv2.imwrite(str(OUT_DIR / img_path.name), img)
print('⇢ 시각화 20장 저장 완료 -', OUT_DIR)
