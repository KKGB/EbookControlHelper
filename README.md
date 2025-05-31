# EbookControlHelper
ebook control helper is a deep learning-powered tool that allows users to navigate eBooks using only their gaze. Leveraging a custom YOLO-based eye-tracking model, this project provides a hands-free, intuitive interface for reading and interacting with digital content.

![Python version](https://img.shields.io/badge/Python-3.9-blue) 

## Installation
```bash
git clone https://github.com/KKGB/EbookControlHelper.git
cd EbookControlHelper

# window
conda env create -f environment_window.yml

# mac
conda env create -f environment_mac.yml
```

## Data
We use a publicly available dataset from [AI Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=548), provided by the Korean government’s AI Hub project.  
The dataset was downloaded for research purposes only and is used in accordance with the terms and conditions specified by AI Hub.

## Model
| Name | Parameters | Link |
|-----------|--------------|-----------|
| YOLOv11s  | 10,083,836        | [Download](https://drive.google.com/file/d/1eqGQkjUDku4U0In6x5GGQciVW7Dm1yHZ/view?usp=sharing) |
| YOLOv11n  | 2,843,388        | [Download](https://drive.google.com/file/d/1kFgXxgROzXhJwsZg7eyFz_wn69kgXu3v/view?usp=sharing) |

## Program
- [Window Download](https://drive.google.com/file/d/1gjHDuqmw1r13LP9Ub-A2x2DA4dHFuWf-/view?usp=drive_link)
- [~~Mac Download~~]()
