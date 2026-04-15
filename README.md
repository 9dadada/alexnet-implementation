# AlexNet Implementation

PyTorch로 구현한 AlexNet 논문 재현 프로젝트입니다.

## Project Structure

```
alexnet-implementation/
├── model/
│   └── alexnet.py       # AlexNet 모델 아키텍처
├── train.py             # CIFAR-10 학습 스크립트
├── inference.py         # 이미지 분류 추론
├── requirements.txt     # 의존성
└── results/             # 학습 결과
```

## Setup

```bash
pip install -r requirements.txt
```

## Reference

- [ImageNet Classification with Deep Convolutional Neural Networks (2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
