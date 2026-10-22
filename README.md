# JENGA_6DOF

젠가 블록 (n층, m번째)의 기준 자세 대비 6DOF 편차를 Gemini 305 RGB-D로 추정한다.

진행 과정, 정확도, 문제/해결, 향후 계획: [docs/SESSION_REPORT.md](docs/SESSION_REPORT.md) (HTML 버전: `docs/SESSION_REPORT.html`)

## 구조

```
config/tower.yaml          기준 타워 정의 (블록 치수, 층 수, 좌표/번호 규칙, 태그)
jenga6dof/tower_model.py   (n, m) -> 타워 좌표계 기준 4x4 nominal pose
jenga6dof/camera.py        Gemini 305 래퍼 (BGR + color 정렬 depth[m])
tools/make_tag_pdf.py      AprilTag 띠 PDF 생성 -> tags/apriltag_36h11_strips.pdf
tools/stream_check.py      M0: 실시간 RGB/depth 확인, 태그 검출 확인, 스냅샷 저장 (인자: 저장 폴더)
tools/calib_capture.py     M1-1: 모서리 뷰 자동 저장 (태그 맵용)
tools/solve_tagmap.py      M1-1: 태그 맵 BA -> config/tagmap.yaml
tools/fit_tower_ref.py     M1-2: 기준 타워 높이/블록 높이 fit -> config/tower_ref.yaml
jenga6dof/tagmap.py        태그 띠 기하, 검출, 태그 맵 BA
jenga6dof/localize.py      태그 맵으로 카메라 위치 추정, 태그맵->타워 좌표 변환
jenga6dof/estimator.py     블록 6DOF 편차 추정 (태그 -> 주변 층 국소 보정 -> 블록 fit)
jenga6dof/ros_pub.py       ROS2 발행 (/jenga/block_pose, /jenga/block_nominal_pose, /jenga/block_delta)
tools/live_block.py        M2: 실시간 (n, m) 블록 편차 표시 + ROS2 발행
tools/eval_ref.py          기준(정리된) 타워 캡처에서 추정 노이즈/편향 평가
```

## 환경

- Python 3.10 venv (`.venv`, `--system-site-packages`로 ROS2 Humble rclpy 사용)
- `.venv/bin/pip install -r requirements.txt`
- 카메라 USB 권한 (최초 1회, sudo 필요):

```bash
sudo .venv/lib/python3.10/site-packages/pyorbbecsdk/shared/install_udev_rules.sh
# 그 다음 카메라 USB 뽑았다 다시 꽂기
```

## 실행

```bash
.venv/bin/python tools/stream_check.py   # s: 스냅샷, q: 종료
```

## 실시간 측정 (M2)

```bash
.venv/bin/python tools/live_block.py            # --no-ros, --no-edges 옵션
# 터미널에 "n m" 입력 (예: 8 2), 창에서 c = 평균 초기화, s = 저장, q = 종료
```

- 결과 delta는 블록 좌표계 기준: x = 블록 긴 방향, y = 폭 방향, z = 위.
  - 홀수층: 블록 +x = 타워 오른쪽(RIGHT), 짝수층: 블록 +x = 타워 뒤쪽(BACK)
- `/jenga/block_delta` = [n, m, dx, dy, dz (mm), roll, pitch, yaw (deg), 샘플 수, valid]

## 좌표/번호 규칙

- 앞면 = 태그 id 0,1 띠가 붙은 지그 박스 면. 위에서 봤을 때 시계방향으로 face 1(id 2,3), 2(id 4,5), 3(id 6,7).
- n = 1(맨 아래) .. 17. 홀수층은 앞에서 긴 면이 보이고 m=1이 가장 앞쪽. 짝수층은 앞에서 끝면이 보이고 m=1이 왼쪽.
- 타워 좌표계: 원점 = 타워 중심축 & 1층 바닥면, +x 오른쪽(앞에서 봤을 때), +y 뒤쪽, +z 위.

## 진행 상황

- [x] 태그 PDF, venv/SDK 설치, 기준 타워 모델
- [x] M0: udev 설치, RGB/depth/태그 검출 확인 (왜곡 모델 8계수)
- [x] M1: 태그 맵 (RMS 0.67px) + 기준 타워 fit (1층 바닥 = 띠 중심 +27.0mm, 블록 높이 22.36mm)
- [x] M2: (n, m) 블록 편차 추정 + 화면 표시 + ROS2 발행 (손 촬영 단계 마무리)
  - 8층 2번 당기기 테스트 (2026-10-22, Close Range High Accuracy 프리셋, depth 1280x800):
    REL dx 0 / -4.5 / -9.8 mm (실제 0 / 5 / 10), yaw/roll/pitch ±2도 이내. ABS는 ±3mm 수준 (손 태그 오차)
  - 9층 1번 yaw 테스트: 오른쪽 끝 5mm -> -2.0도 (기대 -2.5), 10mm 재측정 -> -4.4~-5.5도 (기대 -5.0)
  - 커버리지 (정리 상태): 2-2 / 16-2 / 17-1 모두 REL ±2mm, ±2도 이내.
    ABS는 16-17층에서 앞/뒤 시점별로 ±4~7mm (손 태그 거리 40~45cm 한계) -> 손목 장착(M3)에서 해결 예정
- [ ] M3: 손목 장착 (FK + hand-eye)
