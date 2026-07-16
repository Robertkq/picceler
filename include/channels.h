#pragma once

#include <cstdint>

namespace picceler {

enum class Channel : uint8_t {
  R = 0,
  G = 1,
  B = 2,
  A = 3,
  Count = 4
};

} // namespace picceler