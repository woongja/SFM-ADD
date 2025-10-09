1. Project Overview

SFM-ADD (Shared-Feature Multi-Backend Architecture for Robust Audio Deepfake Detection)
본 프로젝트는 XLS-R 기반 공통 feature extractor를 사용하고,
AASIST와 Conformer-TCM 두 가지 backend를 결합하여
노이즈 환경에서도 견고한 딥페이크 음성 탐지를 목표로 합니다.

구조적으로:

XLS-R(300M) → Shared Embedding Extractor

AASIST + Conformer-TCM → Dual Backend

Score Fusion + Joint Loss Training
