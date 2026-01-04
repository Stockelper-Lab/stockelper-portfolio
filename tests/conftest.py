import sys
from pathlib import Path


# `src/main.py`는 `from routers...` 형태의 import를 사용하므로,
# 테스트 실행 시 `src/`를 PYTHONPATH에 추가해 동일한 import 해석이 되도록 합니다.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


