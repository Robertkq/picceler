#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// Benchmark parameters
constexpr int kBrightnessDelta = 30;
constexpr int kGaussianRadius = 6;
constexpr double kSharpenStrength = 40.0;

struct BenchResult {
  std::string operation;
  std::string label;
  double naiveMs;
  double opencvMs;
};

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  size_t n = samples.size();
  return (n % 2 == 0) ? (samples[n / 2 - 1] + samples[n / 2]) / 2.0 : samples[n / 2];
}

template <typename Fn> double timeMedianMs(int iterations, Fn &&fn) {
  fn(); // untimed warm-up

  std::vector<double> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    auto start = Clock::now();
    fn();
    auto end = Clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  return median(std::move(samples));
}

std::vector<double> buildGaussianKernel(int radius) {
  int size = 2 * radius + 1;
  double sigma = std::max(radius / 2.0, 0.5);

  std::vector<double> values;
  values.reserve(static_cast<size_t>(size) * size);
  double sum = 0.0;
  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      double exponent = -static_cast<double>(x * x + y * y) / (2 * sigma * sigma);
      double value = std::exp(exponent) / (2 * std::numbers::pi * sigma * sigma);
      values.push_back(value);
      sum += value;
    }
  }
  for (double &v : values) {
    v /= sum;
  }
  return values;
}

std::vector<double> buildSharpenKernel(double strengthValue) {
  double strength = strengthValue / 25.0;
  double center = 1.0 + 4.0 * strength;
  double neighbor = -strength;
  return {0.0, neighbor, 0.0, neighbor, center, neighbor, 0.0, neighbor, 0.0};
}

// ---- naive scalar reference implementations. Images are RGBA8 (CV_8UC4).
// All four ops leave the alpha channel untouched

void naiveInvert(const cv::Mat &src, cv::Mat &dst) {
  dst.create(src.size(), src.type());
  for (int y = 0; y < src.rows; ++y) {
    const uint8_t *s = src.ptr<uint8_t>(y);
    uint8_t *d = dst.ptr<uint8_t>(y);
    for (int x = 0; x < src.cols; ++x) {
      d[x * 4 + 0] = 255 - s[x * 4 + 0];
      d[x * 4 + 1] = 255 - s[x * 4 + 1];
      d[x * 4 + 2] = 255 - s[x * 4 + 2];
      d[x * 4 + 3] = s[x * 4 + 3];
    }
  }
}

void naiveBrightness(const cv::Mat &src, cv::Mat &dst, int delta) {
  dst.create(src.size(), src.type());
  for (int y = 0; y < src.rows; ++y) {
    const uint8_t *s = src.ptr<uint8_t>(y);
    uint8_t *d = dst.ptr<uint8_t>(y);
    for (int x = 0; x < src.cols; ++x) {
      for (int c = 0; c < 3; ++c) {
        int v = static_cast<int>(s[x * 4 + c]) + delta;
        d[x * 4 + c] = static_cast<uint8_t>(std::clamp(v, 0, 255));
      }
      d[x * 4 + 3] = s[x * 4 + 3];
    }
  }
}

// Generic RGB-channel 2D convolution used by both gaussian_blur and sharpen.
void naiveConvolveRGB(const cv::Mat &src, cv::Mat &dst, const std::vector<double> &kernel, int kRows, int kCols) {
  dst.create(src.size(), src.type());
  int rowRadius = kRows / 2;
  int colRadius = kCols / 2;

  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      double acc[3] = {0.0, 0.0, 0.0};
      for (int ky = 0; ky < kRows; ++ky) {
        int sy = y + ky - rowRadius;
        if (sy < 0 || sy >= src.rows)
          continue;
        for (int kx = 0; kx < kCols; ++kx) {
          int sx = x + kx - colRadius;
          if (sx < 0 || sx >= src.cols)
            continue;
          double w = kernel[ky * kCols + kx];
          const uint8_t *p = src.ptr<uint8_t>(sy) + sx * 4;
          acc[0] += p[0] * w;
          acc[1] += p[1] * w;
          acc[2] += p[2] * w;
        }
      }
      uint8_t *d = dst.ptr<uint8_t>(y) + x * 4;
      for (int c = 0; c < 3; ++c) {
        d[c] = static_cast<uint8_t>(std::clamp(acc[c], 0.0, 255.0));
      }
      d[3] = src.ptr<uint8_t>(y)[x * 4 + 3];
    }
  }
}

// ---- OpenCV reference implementations ----

void opencvInvert(const cv::Mat &src, cv::Mat &dst) { cv::bitwise_not(src, dst); }

void opencvBrightness(const cv::Mat &src, cv::Mat &dst, int delta) {
  cv::add(src, cv::Scalar(delta, delta, delta, 0), dst);
}

// Splits off the alpha channel so blur/sharpen only do RGB work, matching the
// naive/picceler columns (which also leave alpha untouched) rather than
// having OpenCV do a 4th channel of work the other two columns don't.
cv::Mat splitAlpha(const cv::Mat &rgba, cv::Mat &alphaOut) {
  std::vector<cv::Mat> channels;
  cv::split(rgba, channels);
  alphaOut = channels[3];
  cv::Mat rgb;
  cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, rgb);
  return rgb;
}

cv::Mat mergeAlpha(const cv::Mat &rgb, const cv::Mat &alpha) {
  std::vector<cv::Mat> channels;
  cv::split(rgb, channels);
  channels.push_back(alpha);
  cv::Mat rgba;
  cv::merge(channels, rgba);
  return rgba;
}

void opencvGaussianBlur(const cv::Mat &src, cv::Mat &dst, int radius) {
  cv::Mat alpha;
  cv::Mat rgb = splitAlpha(src, alpha);

  int size = 2 * radius + 1;
  double sigma = std::max(radius / 2.0, 0.5);
  cv::Mat blurred;
  // BORDER_CONSTANT (zero-padding) to match picceler's own border handling,
  // so the two columns differ in speed, not in what they compute at edges.
  cv::GaussianBlur(rgb, blurred, cv::Size(size, size), sigma, sigma, cv::BORDER_CONSTANT);

  dst = mergeAlpha(blurred, alpha);
}

void opencvSharpen(const cv::Mat &src, cv::Mat &dst, double strengthValue) {
  cv::Mat alpha;
  cv::Mat rgb = splitAlpha(src, alpha);

  std::vector<double> kernelValues = buildSharpenKernel(strengthValue);
  cv::Mat kernel(3, 3, CV_64F, kernelValues.data());
  cv::Mat sharpened;
  cv::filter2D(rgb, sharpened, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);

  dst = mergeAlpha(sharpened, alpha);
}

void printTable(const std::vector<BenchResult> &results) {
  std::cout << std::left << std::setw(20) << "operation" << std::right << std::setw(14) << "naive (ms)" << std::setw(14)
            << "opencv (ms)" << "\n";
  std::cout << std::string(48, '-') << "\n";
  std::cout << std::fixed << std::setprecision(2);
  for (const auto &r : results) {
    std::cout << std::left << std::setw(20) << r.label << std::right << std::setw(14) << r.naiveMs << std::setw(14)
              << r.opencvMs << "\n";
  }
}

void printJson(const std::vector<BenchResult> &results, int iterations) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "{\n  \"iterations\": " << iterations << ",\n  \"results\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &r = results[i];
    std::cout << "    {\"operation\": \"" << r.operation << "\", \"label\": \"" << r.label
              << "\", \"naive_ms\": " << r.naiveMs << ", \"opencv_ms\": " << r.opencvMs << "}";
    std::cout << (i + 1 < results.size() ? ",\n" : "\n");
  }
  std::cout << "  ]\n}\n";
}

} // namespace

int main(int argc, char **argv) {
  std::vector<std::string> args(argv + 1, argv + argc);

  std::string imagePath;
  bool haveImagePath = false;
  int iterations = 20;
  bool haveIterations = false;
  bool jsonOutput = false;

  for (const auto &arg : args) {
    if (arg == "--json") {
      jsonOutput = true;
    } else if (!haveImagePath) {
      imagePath = arg;
      haveImagePath = true;
    } else if (!haveIterations) {
      iterations = std::atoi(arg.c_str());
      haveIterations = true;
    } else {
      std::cerr << "reference_bench: unexpected argument '" << arg << "'\n";
      return 1;
    }
  }

  if (!haveImagePath) {
    std::cerr << "usage: reference_bench <image-path> [iterations] [--json]\n";
    return 1;
  }
  if (iterations < 1) {
    std::cerr << "reference_bench: iterations must be >= 1\n";
    return 1;
  }

  cv::Mat loaded = cv::imread(imagePath);
  if (loaded.empty()) {
    std::cerr << "reference_bench: failed to load image: " << imagePath << "\n";
    return 1;
  }

  // Match lib/src/image.cpp's loadImage(): picceler-generated code always
  // operates on RGBA8, so that's the workload naive/OpenCV must match too.
  cv::Mat rgba;
  cv::cvtColor(loaded, rgba, cv::COLOR_BGR2RGBA);

  std::vector<BenchResult> results;

  {
    cv::Mat dst;
    double naiveMs = timeMedianMs(iterations, [&] { naiveInvert(rgba, dst); });
    double opencvMs = timeMedianMs(iterations, [&] { opencvInvert(rgba, dst); });
    results.push_back({"invert", "invert", naiveMs, opencvMs});
  }
  {
    cv::Mat dst;
    double naiveMs = timeMedianMs(iterations, [&] { naiveBrightness(rgba, dst, kBrightnessDelta); });
    double opencvMs = timeMedianMs(iterations, [&] { opencvBrightness(rgba, dst, kBrightnessDelta); });
    results.push_back({"brightness", "brightness(+" + std::to_string(kBrightnessDelta) + ")", naiveMs, opencvMs});
  }
  {
    cv::Mat dst;
    std::vector<double> kernel = buildGaussianKernel(kGaussianRadius);
    int size = 2 * kGaussianRadius + 1;
    double naiveMs = timeMedianMs(iterations, [&] { naiveConvolveRGB(rgba, dst, kernel, size, size); });
    double opencvMs = timeMedianMs(iterations, [&] { opencvGaussianBlur(rgba, dst, kGaussianRadius); });
    results.push_back({"gaussian_blur", "gaussian_blur(r=" + std::to_string(kGaussianRadius) + ")", naiveMs, opencvMs});
  }
  {
    cv::Mat dst;
    std::vector<double> kernel = buildSharpenKernel(kSharpenStrength);
    double naiveMs = timeMedianMs(iterations, [&] { naiveConvolveRGB(rgba, dst, kernel, 3, 3); });
    double opencvMs = timeMedianMs(iterations, [&] { opencvSharpen(rgba, dst, kSharpenStrength); });
    results.push_back({"sharpen", "sharpen(3x3)", naiveMs, opencvMs});
  }

  if (jsonOutput) {
    printJson(results, iterations);
  } else {
    printTable(results);
  }

  return 0;
}
