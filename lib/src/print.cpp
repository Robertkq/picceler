#include "runtime.h"

#include <print>

void piccelerPrintString(const char *str) { std::print(std::cout, "{}", str); }

void piccelerPrintFloat64(double value) { std::print(std::cout, "{}", value); }