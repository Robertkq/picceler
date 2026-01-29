#include "utils.h"

namespace picceler::utils {

std::string expandTilde(const std::string& inputPath) {
    if (inputPath.empty() || inputPath[0] != '~') {
        return inputPath;
    }

    std::filesystem::path homePath;

#ifdef _WIN32
    const char* userProfile = std::getenv("USERPROFILE");
    if (!userProfile) {
        const char* hd = std::getenv("HOMEDRIVE");
        const char* hp = std::getenv("HOMEPATH");
        if (hd && hp) {
            homePath = std::string(hd) + std::string(hp);
        } else {
            return inputPath;
        }
    } else {
        homePath = userProfile;
    }
#else
    const char* homeEnv = std::getenv("HOME");
    if (homeEnv) {
        homePath = homeEnv;
    } else {
        struct passwd* pw = getpwuid(getuid());
        if (pw) {
            homePath = pw->pw_dir;
        } else {
            return inputPath;
        }
    }
#endif
    std::string remainder = inputPath.substr(1);

    if (!remainder.empty() && (remainder[0] == '/' || remainder[0] == '\\')) {
        remainder = remainder.substr(1);
    }

    homePath /= remainder;

    return homePath.string();
}

} // namespace picceler::utils
