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

/**
 * Expands a leading tilde in a filesystem path to the current user's home directory.
 *
 * On POSIX systems, this uses the HOME environment variable when available and
 * falls back to getpwuid(getuid()) if HOME is not set. On Windows, it uses the
 * USERPROFILE environment variable, or HOMEDRIVE + HOMEPATH if USERPROFILE is
 * not available.
 *
 * If the input does not begin with '~', or if the home directory cannot be
 * determined (e.g., required environment variables are unset and system calls
 * fail), the original inputPath is returned unchanged.
 *
 * @param inputPath Path string that may start with '~' to refer to the user's home.
 * @return A string where a leading '~' has been replaced by the user's home
 *         directory, or the original inputPath if no expansion is performed.
 */
std::string expandTilde(const std::string& inputPath);

} // namespace picceler::utils
