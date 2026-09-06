# Benchmark Results

- Date: 2026-09-06
- CPU: Intel(R) Core(TM) i5-14600KF
- Image: `bench.jpg` (1092344 bytes)
- Iterations: 10

Generated locally by `bench/run_bench.py`, not in CI -- see bench/reference_bench.cpp for the naive/OpenCV implementations and bench/pic/ for the picceler programs.

| operation | picceler | naive C++ | OpenCV |
| --- | --- | --- | --- |
| invert | 18.6 ms | 3.1 ms | 1.5 ms |
| brightness(+30) | 17.8 ms | 3.9 ms | 1.7 ms |
| gaussian_blur(r=6) | 1411.3 ms | 1044.2 ms | 7.0 ms |
| sharpen(3x3) | 93.3 ms | 74.6 ms | 10.4 ms |
