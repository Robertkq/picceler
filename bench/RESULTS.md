# Benchmark Results

- Date: 2026-09-06
- CPU: Intel(R) Core(TM) i5-14600KF
- Image: `bench.jpg` (1092344 bytes)
- Iterations: 50

Generated locally by `bench/run_bench.py`, not in CI -- see bench/reference_bench.cpp for the naive/OpenCV implementations and bench/pic/ for the picceler programs.

| operation | picceler | naive C++ | OpenCV |
| --- | --- | --- | --- |
| invert | 18.6 ms | 7.0 ms | 4.8 ms |
| brightness(+30) | 18.6 ms | 7.8 ms | 5.1 ms |
| gaussian_blur(r=6) | 1199.0 ms | 1070.8 ms | 12.5 ms |
| sharpen(3x3) | 74.0 ms | 81.6 ms | 15.7 ms |
