#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include "spdlog/cfg/env.h"

int main(int argc, char **argv) {
  spdlog::cfg::load_env_levels();
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}