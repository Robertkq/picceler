# Benchmark Results

- Date: 2026-09-06
- CPU: Intel(R) Core(TM) i5-14600KF
- Image: `bench.jpg` (1092344 bytes)
- Iterations: 50

Generated locally by `bench/run_bench.py`, not in CI -- see bench/reference_bench.cpp for the naive/OpenCV implementations and bench/pic/ for the picceler programs.

| operation | picceler | naive C++ | OpenCV |
| --- | --- | --- | --- |
| invert | 18.4 ms | 6.8 ms | 4.9 ms |
| brightness(+30) | 18.1 ms | 8.0 ms | 5.3 ms |
| gaussian_blur(r=6) | 1415.7 ms | 1051.5 ms | 12.3 ms |
| sharpen(3x3) | 93.4 ms | 79.4 ms | 15.4 ms |
