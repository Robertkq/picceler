#include "lld/Common/Driver.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include <vector>
#include <string>

LLD_HAS_DRIVER(elf)

int main(int argc, char **argv) {
  // Initialize LLVM (sets up signal handlers, command-line parsing support, etc.)
  llvm::InitLLVM init(argc, argv);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  std::vector<const char *> args;
  args.reserve(argc);
  for (int i = 0; i < argc; ++i)
    args.push_back(argv[i]);

  // Provide drivers array with only the ELF (GNU) driver
  static const lld::DriverDef drivers[] = {{lld::Gnu, &lld::elf::link}};

  if (!args.empty()) {
    args[0] = "ld.lld";
  }
  lld::Result res = lld::lldMain(llvm::ArrayRef<const char *>(args.data(), args.size()), llvm::outs(), llvm::errs(), llvm::ArrayRef<lld::DriverDef>(drivers));
  return res.retCode;
}
