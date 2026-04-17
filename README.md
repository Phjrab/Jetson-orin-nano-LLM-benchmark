# Local LLM Benchmark (Jetson Orin Nano)

Jetson Orin Nano(8GB)에서 GGUF 모델을 로컬 추론하고, 모델별 성능을 같은 조건으로 비교하는 프로젝트입니다.

기본 엔진은 `llama-cpp-python`(CUDA 빌드)입니다.

## 1) Environment 생성

```bash
mkdir -p local_llm_bench
cd local_llm_bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 2) Jetson 빌드 의존성 설치

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3-dev libopenblas-dev
```

## 3) Python 패키지 설치

```bash
pip install -r requirements.txt
```

`llama-cpp-python` CUDA 빌드:

```bash
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_BUILD_TYPE=Release" FORCE_CMAKE=1 pip install --no-binary=:all: llama-cpp-python
```

구버전 빌드 플래그가 필요한 경우:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_BUILD_TYPE=Release" FORCE_CMAKE=1 pip install --no-binary=:all: llama-cpp-python
```

## 4) Llama 외 모델 설치(다운로드) 예시

아래는 Jetson 8GB에서 비교하기 좋은 경량/양자화 모델 예시입니다.

```bash
mkdir -p models/qwen2.5-3b models/gemma2-2b models/phi3-mini

hf download bartowski/Qwen2.5-3B-Instruct-GGUF \
  --include "Qwen2.5-3B-Instruct-Q4_K_M.gguf" \
  --local-dir models/qwen2.5-3b

hf download bartowski/gemma-2-2b-it-GGUF \
  --include "gemma-2-2b-it-Q4_K_M.gguf" \
  --local-dir models/gemma2-2b

hf download bartowski/Phi-3-mini-4k-instruct-GGUF \
  --include "Phi-3-mini-4k-instruct-Q4_K_M.gguf" \
  --local-dir models/phi3-mini
```

주의: 모델 파일명은 리포지토리 업데이트로 바뀔 수 있으니, 필요하면 `--include` 패턴을 조정하세요.
필요 시 먼저 로그인:

```bash
hf auth login
```

## 5) 단일 모델 벤치마크

```bash
python bench/benchmark.py \
  --model /absolute/path/to/model.gguf \
  --model-name qwen2.5-3b-q4 \
  --num-prompts 3 \
  --max-tokens 128 \
  --n-ctx 1024 \
  --n-gpu-layers 35 \
  --warmup
```

주의: 줄바꿈으로 실행할 때는 각 줄 끝에 `\`가 필요합니다. 아니면 한 줄로 실행하세요.

출력 지표:
- Model Load Time
- TTFT (Time to First Token)
- TPS (Tokens/s)
- 프로세스 메모리(RSS/Peak RSS)

## 6) 여러 모델 자동 비교

모델 목록 CSV 예시 파일: `models.example.csv`

```bash
python bench/compare_models.py \
  --models-file models.example.csv \
  --num-prompts 3 \
  --max-tokens 128 \
  --n-ctx 1024 \
  --n-gpu-layers 35 \
  --warmup \
  --output-csv outputs/comparison_results.csv
```

결과:
- 콘솔에 모델별 TTFT/TPS/Peak RSS 요약
- `outputs/comparison_results.csv`에 전체 비교 결과 저장

### 원클릭 실행 스크립트(권장)

Jetson 메모리 상태에 따라 모델별로 안전한 `n_gpu_layers`를 자동 탐색한 뒤,
비교/그래프/랭킹까지 한 번에 실행합니다.

기본적으로 스크립트 시작 시 `nvpmodel`/`jetson_clocks`를 사용해 Max Power를 적용하고,
종료 시 이전 상태로 복원합니다.

```bash
chmod +x scripts/run_all_bench.sh
./scripts/run_all_bench.sh
```

환경변수로 파라미터를 바꿀 수 있습니다:

```bash
NUM_PROMPTS=3 MAX_TOKENS=128 N_CTX=1024 ./scripts/run_all_bench.sh
```

전원 모드 자동 제어 관련 옵션:

```bash
# Max Power 자동 적용 끄기
AUTO_MAX_POWER=0 ./scripts/run_all_bench.sh

# 종료 시 원복 끄기
RESTORE_POWER_ON_EXIT=0 ./scripts/run_all_bench.sh

# 특정 nvpmodel ID를 강제로 사용 (기본: auto = MAXN 계열 자동 탐지)
MAX_POWER_MODE=2 ./scripts/run_all_bench.sh
```

`GPU_LAYER_MARGIN`으로 자동 탐색된 레이어에서 안전 마진을 줄 수 있습니다(기본 2):

```bash
GPU_LAYER_MARGIN=3 ./scripts/run_all_bench.sh
```

탐색 검증 토큰 길이를 본실행과 맞추려면 `PROBE_MAX_TOKENS`를 사용하세요:

```bash
MAX_TOKENS=128 PROBE_MAX_TOKENS=128 ./scripts/run_all_bench.sh
```

`matplotlib`가 설치되어 있지 않으면 CSV/랭킹은 생성하고 PNG 그래프 단계는 자동으로 건너뜁니다.

생성 파일:
- `outputs/comparison_results.csv`
- `outputs/comparison_ranked.csv`
- `outputs/comparison_results.png`

하드웨어 중심 정렬/랭킹 예시:

```bash
PLOT_SORT_BY=avg_power_w \
RANK_W_TPS=0.35 RANK_W_RSS=0.20 RANK_W_TTFT=0.15 \
RANK_W_POWER=0.15 RANK_W_GPU_TEMP=0.10 RANK_W_CPU_TEMP=0.05 \
./scripts/run_all_bench.sh
```

## 7) 결과 그래프 자동 생성

```bash
python bench/plot_results.py \
  --input-csv outputs/comparison_results.csv \
  --output-png outputs/comparison_results.png \
  --sort-by avg_tps
```

생성 파일:
- `outputs/comparison_results.png` (기본 성능 지표 + CSV에 값이 있는 Jetson 하드웨어 지표 자동 포함)

정렬 가능한 컬럼 예시:
- `avg_tps`, `avg_ttft_s`, `peak_process_rss_mb`, `model_load_time_s`
- `avg_gpu_util_pct`, `avg_power_w`, `peak_ram_vram_used_mb`, `peak_gpu_temp_c`, `peak_cpu_temp_c`

그래프가 안 뜨면 아래로 설치 확인:

```bash
pip install matplotlib
```

## 8) 모델 자동 랭킹(가중치 점수)

기본 가중치:
- TPS: 0.50 (높을수록 좋음)
- Peak RSS: 0.30 (낮을수록 좋음)
- TTFT: 0.20 (낮을수록 좋음)

선택 하드웨어 가중치(기본 0.0):
- GPU Util: `--w-gpu-util` (높을수록 좋음)
- Power: `--w-power` (낮을수록 좋음)
- Peak Unified RAM/VRAM: `--w-jetson-ram` (낮을수록 좋음)
- Peak GPU Temp: `--w-gpu-temp` (낮을수록 좋음)
- Peak CPU Temp: `--w-cpu-temp` (낮을수록 좋음)

```bash
python bench/rank_models.py \
  --input-csv outputs/comparison_results.csv \
  --output-csv outputs/comparison_ranked.csv \
  --w-tps 0.50 \
  --w-rss 0.30 \
  --w-ttft 0.20 \
  --w-power 0.00 \
  --w-gpu-temp 0.00 \
  --w-cpu-temp 0.00
```

생성 파일:
- `outputs/comparison_ranked.csv`

## 9) GitHub 리포지토리 연결

아직 로컬에 Git 초기화가 안 되어 있다면:

```bash
git init
git add .
git commit -m "Initial local LLM benchmark project"
git branch -M main
git remote add origin https://github.com/Phjrab/Jetson-orin-nano-LLM-benchmark.git
git push -u origin main
```

이미 원격이 등록되어 있으면 URL만 교체:

```bash
git remote set-url origin https://github.com/Phjrab/Jetson-orin-nano-LLM-benchmark.git
git push -u origin main
```

## 10) Jetson 모니터링 팁

```bash
sudo -H pip3 install -U jetson-stats
jtop
```

벤치마크 실행 중 `jtop`에서 RAM/SWAP/GPU 사용량을 같이 확인하면, 8GB 공유 메모리 병목을 더 정확히 파악할 수 있습니다.

## 11) 8GB 메모리 최적화 팁

OOM이 나면 아래 순서대로 줄이세요.

1. `--n-gpu-layers`
2. `--n-ctx`
3. `--max-tokens`

추가 팁:
- 먼저 `Q4_K_M`로 시작하고, 안정화 후 `Q5_K_M`로 올려 품질/속도를 재평가하세요.
- 모델 비교 시 프롬프트/파라미터(temperature, top-p, max-tokens)는 고정해야 공정합니다.

## 12) Web Chat 서버 실행/종료

웹 UI 서버 실행:

```bash
chmod +x scripts/start_server.sh scripts/stop_server.sh
./scripts/start_server.sh
```

웹 UI 서버 종료:

```bash
./scripts/stop_server.sh
```

기본 주소는 `http://0.0.0.0:8000`이며, 환경변수로 변경할 수 있습니다:

```bash
HOST=0.0.0.0 PORT=8001 ./scripts/start_server.sh
```

모델 로드 시 Jetson OOM이 발생하면 `web/app.py`가 자동으로 `n_gpu_layers`와 `n_ctx`를 낮춰 재시도합니다.
