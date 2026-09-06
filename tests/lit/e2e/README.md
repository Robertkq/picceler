# E2E tests

Each test compiles a `.pic` file, runs the binary, and checks stdout text.

This proves the program compiles, links, and runs. It does not check pixel
output: a `gaussian_blur` that returned its input unchanged would still pass.

Pixel correctness is not tested here.
