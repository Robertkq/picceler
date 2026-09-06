# Benchmark Results

- Date: 2026-09-06
- CPU: Intel(R) Core(TM) i5-14600KF
- Image: `bench.jpg` (1092344 bytes)
- Iterations: 10

Generated locally by `bench/run_bench.py`, not in CI -- see bench/reference_bench.cpp for the naive/OpenCV implementations and bench/pic/ for the picceler programs.

| operation | picceler | naive C++ | OpenCV |
| --- | --- | --- | --- |
| invert | 18.3 ms | 3.0 ms | 1.5 ms |
| brightness(+30) | 17.9 ms | 3.9 ms | 1.6 ms |
| gaussian_blur(r=6) | 1412.0 ms | 1033.2 ms | 7.0 ms |
| sharpen(3x3) | 92.6 ms | 74.7 ms | 10.5 ms |