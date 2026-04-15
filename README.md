# AlexNet Implementation

PyTorch로 구현한 AlexNet 논문 재현 프로젝트입니다.  
CIFAR-10 데이터셋을 사용하여 이미지 분류를 수행합니다.

## Project Structure

```
alexnet-implementation/
├── model/
│   └── alexnet.py              # AlexNet 모델 아키텍처
├── train.py                    # CIFAR-10 학습 스크립트 (로컬용)
├── inference.py                # 이미지 분류 추론
├── notebooks/
│   └── alexnet_train.ipynb     # Google Colab 학습 노트북
├── requirements.txt            # 의존성
└── results/                    # 학습 결과 (그래프, 모델 가중치)
```

## Quick Start (Google Colab)

GPU 환경이 없다면 Colab에서 바로 실행할 수 있습니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/9dadada/alexnet-implementation/blob/main/notebooks/alexnet_train.ipynb)

1. 위 버튼 클릭
2. Runtime → Change runtime type → **T4 GPU** 선택
3. 셀 순서대로 실행

## Local Setup

```bash
pip install -r requirements.txt
python train.py
```

## Model Architecture

| Layer | Output Size | Details |
|-------|-------------|---------|
| Conv1 | 55x55x96 | kernel=11, stride=4, ReLU, LRN, MaxPool |
| Conv2 | 13x13x256 | kernel=5, ReLU, LRN, MaxPool |
| Conv3 | 13x13x384 | kernel=3, ReLU |
| Conv4 | 13x13x384 | kernel=3, ReLU |
| Conv5 | 6x6x256 | kernel=3, ReLU, MaxPool |
| FC1 | 4096 | Dropout(0.5), ReLU |
| FC2 | 4096 | Dropout(0.5), ReLU |
| FC3 | 10 | Output (CIFAR-10) |

## Reference

- [ImageNet Classification with Deep Convolutional Neural Networks (2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
