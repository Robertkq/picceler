# Benchmark Results

- Date: 2026-09-06
- CPU: Intel(R) Core(TM) i5-14600KF
- Image: `bench.jpg` (1092344 bytes)
- Iterations: 50

Generated locally by `bench/run_bench.py`, not in CI -- see bench/reference_bench.cpp for the naive/OpenCV implementations and bench/pic/ for the picceler programs.

| operation | picceler | naive C++ | OpenCV |
| --- | --- | --- | --- |
| invert | 18.4 ms | 3.1 ms | 1.6 ms |
| brightness(+30) | 17.8 ms | 3.9 ms | 1.6 ms |
| gaussian_blur(r=6) | 1412.1 ms | 1038.7 ms | 17.8 ms |
| sharpen(3x3) | 92.5 ms | 74.8 ms | 23.3 ms |
