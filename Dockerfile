FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY runtime.py benchmark.py synthesize.py download.py profile_summary.py tests_runtime.py prepare.py ./
COPY experiments ./experiments
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CMD ["python", "benchmark.py", "--steps", "8", "--repeats", "5", "--output", "results/oracle_fp32_8.json"]
