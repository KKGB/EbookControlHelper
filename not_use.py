from pathlib import Path

# 클래스 인덱스 정의
class_names = {
    2: "right_eyelid_open",
    3: "right_eyelid_closed",
    4: "left_eyelid_open",
    5: "left_eyelid_closed"
}

counts = {name: 0 for name in class_names.values()}

# YOLO .txt 경로 설정
root = Path("data/labels/val")
txt_files = list(root.rglob("*.txt"))

# 라벨 파일 탐색
for txt in txt_files:
    with open(txt, "r") as f:
        for line in f:
            if not line.strip():
                continue
            cls_id = int(line.split()[0])
            if cls_id in class_names:
                counts[class_names[cls_id]] += 1

# 출력
for name, cnt in counts.items():
    print(f"{name}: {cnt}")
