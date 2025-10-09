# SFM-ADD: Shared-Feature Multi-Backend Architecture for Robust Audio Deepfake Detection

---

## 🚀 Overview
**SFM-ADD**는 XLS-R(300M) 기반 **공통 특징 추출기(Shared Feature Extractor)** 를 중심으로,  
AASIST와 Conformer-TCM 두 개의 백엔드를 결합하여  
**잡음 환경에서도 견고한 오디오 딥페이크 탐지**를 수행하는 모델입니다.

### 핵심 아이디어
✅ Shared XLS-R Embedding  
✅ AASIST + Conformer-TCM Dual Backend  
✅ Score Fusion + Joint Loss Optimization  

---

### 1️⃣ Conda 환경 생성
```bash
conda create -n sfm python=3.9 -y
conda activate sfm
```

```bash
# ex1)
pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
# ex2)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```
install fairseq:
```bash
pip install "pip<24.1"

git clone https://github.com/facebookresearch/fairseq.git fairseq_dir
cd fairseq_dir
git checkout a54021305d6b3c
pip install --editable ./
```
