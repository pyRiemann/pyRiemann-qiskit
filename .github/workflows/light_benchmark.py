name: Light Benchmark

on:
  push:
    paths:
    - 'pyriemann_qiskit/**'
    - 'examples/**'
    - '.github/workflows/light_benchmark.yml'
  pull_request:
    paths:
    - 'pyriemann_qiskit/**'
    - 'examples/**'
    - '.github/workflows/light_benchmark.yml'

jobs:
  light_benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: light_benchmark.yml
      - name: Install dependencies
        run: |
          apt-get --allow-releaseinfo-change update
          python -m pip install --upgrade pip
          apt-get -y install --fix-missing git-core
          apt-get -y install build-essential
          pip install -r doc/requirements.txt
      - name: Run benchmark script
        run: |
          python benchmarks/light_benchmark.py
