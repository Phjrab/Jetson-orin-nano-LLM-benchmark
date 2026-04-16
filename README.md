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

huggingface-cli download bartowski/Qwen2.5-3B-Instruct-GGUF \
  --include "Qwen2.5-3B-Instruct-Q4_K_M.gguf" \
  --local-dir models/qwen2.5-3b

huggingface-cli download bartowski/gemma-2-2b-it-GGUF \
  --include "gemma-2-2b-it-Q4_K_M.gguf" \
  --local-dir models/gemma2-2b

huggingface-cli download bartowski/Phi-3-mini-4k-instruct-GGUF \
  --include "Phi-3-mini-4k-instruct-Q4_K_M.gguf" \
  --local-dir models/phi3-mini
```

주의: 모델 파일명은 리포지토리 업데이트로 바뀔 수 있으니, 필요하면 `--include` 패턴을 조정하세요.

## 5) 단일 모델 벤치마크

```bash
python benchmark.py \
  --model /absolute/path/to/model.gguf \
  --model-name qwen2.5-3b-q4 \
  --num-prompts 3 \
  --max-tokens 128 \
  --n-ctx 1024 \
  --n-gpu-layers 35 \
  --warmup
```

출력 지표:
- Model Load Time
- TTFT (Time to First Token)
- TPS (Tokens/s)
- 프로세스 메모리(RSS/Peak RSS)

## 6) 여러 모델 자동 비교

모델 목록 CSV 예시 파일: `models.example.csv`

```bash
python compare_models.py \
  --models-file models.example.csv \
  --num-prompts 3 \
  --max-tokens 128 \
  --n-ctx 1024 \
  --n-gpu-layers 35 \
  --warmup \
  --output-csv comparison_results.csv
```

결과:
- 콘솔에 모델별 TTFT/TPS/Peak RSS 요약
- `comparison_results.csv`에 전체 비교 결과 저장

## 7) 결과 그래프 자동 생성

```bash
python plot_results.py \
  --input-csv comparison_results.csv \
  --output-png comparison_results.png \
  --sort-by avg_tps
```

생성 파일:
- `comparison_results.png` (TPS, TTFT, Peak RSS, Load Time 4개 그래프)

그래프가 안 뜨면 아래로 설치 확인:

```bash
pip install matplotlib
```

## 8) 모델 자동 랭킹(가중치 점수)

기본 가중치:
- TPS: 0.50 (높을수록 좋음)
- Peak RSS: 0.30 (낮을수록 좋음)
- TTFT: 0.20 (낮을수록 좋음)

```bash
python rank_models.py \
  --input-csv comparison_results.csv \
  --output-csv comparison_ranked.csv \
  --w-tps 0.50 \
  --w-rss 0.30 \
  --w-ttft 0.20
```

생성 파일:
- `comparison_ranked.csv`

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
