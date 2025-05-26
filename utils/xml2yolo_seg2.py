# xml2yolo_seg.py  — <image> 루프 추가 버전
import xml.etree.ElementTree as ET
from pathlib import Path
import itertools, argparse

CLASS_MAP = {
    'right_iris':0, 'left_iris':1,
    'right_eyelid':2, 'left_eyelid':3
}

def polygon_to_yolo(points, w, h):
    xs, ys = zip(*points)
    cx, cy = (min(xs)+max(xs))/(2*w), (min(ys)+max(ys))/(2*h)
    bw, bh = (max(xs)-min(xs))/w, (max(ys)-min(ys))/h
    poly = list(itertools.chain.from_iterable([(x/w, y/h) for x, y in points]))
    return poly

def convert_xml(xml_path: Path, dst_root: Path):
    root  = ET.parse(xml_path).getroot()

    # ── XML 경로 중 TL 이후 디렉터리( G1/001/30 )만 추출 ──
    parts   = xml_path.parts
    idx     = parts.index('VL') # val 은 VL
    base_rel = list(parts[idx+1:-1])          # ['G1','001','30']

    # ── XML 안의 모든 <image> 태그를 순회 ──
    for img_tag in root.findall('./image'):
        w, h     = float(img_tag.attrib['width']), float(img_tag.attrib['height'])
        img_name = img_tag.attrib['name']                    # *.jpg 이름

        #   출력 txt 절대경로  : …/labels/G1/001/30/RGB/<img>.txt
        rel = base_rel + ['RGB', img_name.replace('.jpg', '.txt')]
        txt_path = dst_root.joinpath(*rel)

        lines = []
        for obj in img_tag:
            label = obj.attrib.get('label')        # ex) 'right_eyelid'
            if label.endswith('_eyelid'):
                # 열린·닫힘 구분 없이 'right_eyelid' 또는 'left_eyelid'로 통일
                label = label                      # 그대로 사용 (status 무시)

            if label not in CLASS_MAP or 'points' not in obj.attrib:
                continue

            pts = [(float(x), float(y))
                   for x, y in (p.split(',') for p in obj.attrib['points'].split(';'))]
            nums = polygon_to_yolo(pts, w, h)
            lines.append(f"{CLASS_MAP[label]} " +
                         " ".join(f"{n:.6f}" for n in nums))

        if lines:                                        # 객체가 하나라도 있을 때만 저장
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text("\n".join(lines))

def walk_and_convert(src_root: Path, dst_root: Path):
    xml_files = list(src_root.rglob('*.xml'))
    for i, xml_f in enumerate(xml_files, 1):
        convert_xml(xml_f, dst_root)
        if i % 500 == 0:
            print(f"{i}/{len(xml_files)} converted")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="labels/train/TL 등 XML 루트")
    ap.add_argument("--dst", required=True, help="YOLO txt 최상위 (labels/train)")
    args = ap.parse_args()
    walk_and_convert(Path(args.src), Path(args.dst))

# python utils/xml2yolo_seg.py     --src data/labels/train/TL     --dst data/labels/train