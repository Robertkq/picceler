#pragma once

#include <filesystem>
#include <string>
#include <cstdlib>

#ifndef _WIN32
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#endif

namespace picceler::utils {

std::string expandTilde(const std::string& inputPath);

} // namespace picceler::utils
