# Benchmark Results

- Date: 2026-09-06
- CPU: Intel(R) Core(TM) i5-14600KF
- Image: `bench.jpg` (1092344 bytes)
- Iterations: 50

Generated locally by `bench/run_bench.py`, not in CI -- see bench/reference_bench.cpp for the naive/OpenCV implementations and bench/pic/ for the picceler programs.

| operation | picceler | naive C++ | OpenCV |
| --- | --- | --- | --- |
| invert | 9.5 ms | 7.0 ms | 4.8 ms |
| brightness(+30) | 9.4 ms | 7.9 ms | 5.2 ms |
| gaussian_blur(r=6) | 1106.1 ms | 1039.5 ms | 12.1 ms |
| sharpen(3x3) | 72.4 ms | 78.8 ms | 15.3 ms |
