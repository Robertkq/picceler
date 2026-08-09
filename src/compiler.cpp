#include "compiler.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

#include "linker_config.h"
#include "lld/Common/Driver.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "spdlog/spdlog.h"
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>


namespace {

constexpr std::array<std::string_view, 6> kDefaultLibrarySearchDirs = {
    "/usr/lib", "/usr/lib64", "/usr/local/lib", "/usr/local/lib64", "/lib", "/lib64"};

std::vector<std::string> splitDelimited(std::string_view input, char delimiter) {
  std::vector<std::string> tokens;
  size_t current = 0;
  while (current < input.size()) {
    const size_t next = input.find(delimiter, current);
    const size_t length = (next == std::string_view::npos) ? input.size() - current : next - current;
    if (length > 0) {
      tokens.emplace_back(input.substr(current, length));
    }
    if (next == std::string_view::npos) {
      break;
    }
    current = next + 1;
  }
  return tokens;
}

std::vector<std::string> splitWhitespace(std::string_view input) {
  std::vector<std::string> tokens;
  std::string current;
  for (const char c : input) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (!current.empty()) {
        tokens.push_back(std::move(current));
        current.clear();
      }
      continue;
    }
    current.push_back(c);
  }
  if (!current.empty()) {
    tokens.push_back(std::move(current));
  }
  return tokens;
}

void appendUnique(std::vector<std::string> &values, const std::string &value) {
  if (value.empty()) {
    return;
  }
  if (std::find(values.begin(), values.end(), value) == values.end()) {
    values.push_back(value);
  }
}

std::string joinForLog(const std::vector<std::string> &values) {
  std::string joined;
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      joined += ", ";
    }
    joined += values[index];
  }
  return joined;
}

std::optional<std::string> findLibraryByName(std::string_view libraryName,
                                             const std::vector<std::string> &searchDirs) {
  if (libraryName.empty()) {
    return std::nullopt;
  }

  const bool hasLibPrefix = libraryName.rfind("lib", 0) == 0;
  std::vector<std::string> candidateNames;
  if (libraryName.find(".so") != std::string_view::npos || libraryName.find(".a") != std::string_view::npos) {
    candidateNames.emplace_back(libraryName);
  } else {
    candidateNames.emplace_back((hasLibPrefix ? "" : "lib") + std::string(libraryName) + ".so");
    candidateNames.emplace_back((hasLibPrefix ? "" : "lib") + std::string(libraryName) + ".a");
  }

  for (const auto &dir : searchDirs) {
    for (const auto &candidateName : candidateNames) {
      const auto candidatePath = std::filesystem::path(dir) / candidateName;
      if (std::filesystem::exists(candidatePath)) {
        return candidatePath.string();
      }
    }
  }

  return std::nullopt;
}

std::vector<std::string> collectGccLibraryDirs() {
  std::vector<std::string> gccLibraryDirs;
  const std::filesystem::path gccRoot("/usr/lib/gcc");
  if (!std::filesystem::exists(gccRoot)) {
    return gccLibraryDirs;
  }

  for (const auto &archEntry : std::filesystem::directory_iterator(gccRoot)) {
    if (!archEntry.is_directory()) {
      continue;
    }

    for (const auto &versionEntry : std::filesystem::directory_iterator(archEntry.path())) {
      if (versionEntry.is_directory()) {
        gccLibraryDirs.push_back(versionEntry.path().string());
      }
    }
  }

  std::sort(gccLibraryDirs.begin(), gccLibraryDirs.end(), std::greater<>());
  return gccLibraryDirs;
}

std::optional<std::string> findStartupObject(std::string_view objectName,
                                             const std::vector<std::string> &searchDirs) {
  for (const auto &dir : searchDirs) {
    const auto candidatePath = std::filesystem::path(dir) / objectName;
    if (std::filesystem::exists(candidatePath)) {
      return candidatePath.string();
    }
  }
  return std::nullopt;
}

std::vector<std::string> collectOpenCvLibraries(std::vector<std::string> &searchDirs, std::string &diagnosticError) {
  std::vector<std::string> resolvedLibraries;
  const auto openCvItems = splitDelimited(PICCELER_OPENCV_LINK_ITEMS, ';');
  for (const auto &item : openCvItems) {
    if (item.empty()) {
      continue;
    }
    if (item.rfind("-L", 0) == 0) {
      appendUnique(searchDirs, item.substr(2));
      continue;
    }
    if (std::filesystem::path(item).is_absolute()) {
      if (!std::filesystem::exists(item)) {
        diagnosticError = "Configured OpenCV library does not exist: " + item;
        return {};
      }
      resolvedLibraries.push_back(item);
      continue;
    }

    std::string libraryToken = item;
    if (libraryToken.rfind("-l", 0) == 0) {
      libraryToken = libraryToken.substr(2);
    }

    auto resolvedLibrary = findLibraryByName(libraryToken, searchDirs);
    if (!resolvedLibrary) {
      diagnosticError =
          "Could not resolve OpenCV link dependency: " + item +
          ". Check your OpenCV installation and PICCELER_OPENCV_LINK_DIRS configuration.";
      return {};
    }
    resolvedLibraries.push_back(*resolvedLibrary);
  }

  return resolvedLibraries;
}

std::vector<std::string> collectImplicitLinkerArgs(const std::vector<std::string> &searchDirs) {
  std::vector<std::string> implicitArgs;

  auto appendTokens = [&implicitArgs](const std::vector<std::string> &tokens) {
    for (const auto &token : tokens) {
      if (token.empty()) {
        continue;
      }
      if (std::find(implicitArgs.begin(), implicitArgs.end(), token) == implicitArgs.end()) {
        implicitArgs.push_back(token);
      }
    }
  };

  appendTokens(splitDelimited(PICCELER_CXX_IMPLICIT_LINK_LIBS, ';'));
  appendTokens(splitWhitespace(PICCELER_CXX_STANDARD_LIBRARIES));

  std::vector<std::string> resolvedArgs;
  for (const auto &arg : implicitArgs) {
    if (arg.rfind("-", 0) == 0) {
      resolvedArgs.push_back(arg);
      continue;
    }

    const auto resolvedLibrary = findLibraryByName(arg, searchDirs);
    if (resolvedLibrary) {
      resolvedArgs.push_back(*resolvedLibrary);
      continue;
    }

    resolvedArgs.push_back("-l" + arg);
  }

  return resolvedArgs;
}

} // namespace

namespace picceler {

Compiler::Compiler()
    : _cliApp("picceler compiler"), _cliOptions(), _lexer(), _parser(), _context(initRegistry()), _mlirGen(&_context),
      _passManager(&_context) {
  _cliApp.add_option("input_file", _cliOptions._inputFile, "Input source file")->required()->check(CLI::ExistingFile);
  _cliApp.add_option("-o,--output", _cliOptions._outputFile, "Output executable file")->default_val("a.out");

  _context.loadAllAvailableDialects();
  spdlog::trace("Initialized MLIR Dialects:");
  for (auto *dialect : _context.getLoadedDialects()) {
    spdlog::trace(" - {}", dialect->getNamespace());
  }
}

mlir::DialectRegistry Compiler::initRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<PiccelerDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  return registry;
}

bool Compiler::run() {
  const auto &inputFile = _cliOptions._inputFile;
  const auto &outputFile = _cliOptions._outputFile;

  auto sourceResult = _lexer.setSource(inputFile);
  if (!sourceResult) {
    spdlog::error("Failed to set source file: {}", sourceResult.error().message());
    return false;
  }

  auto tokensResult = _lexer.getTokens();
  if (!tokensResult) {
    spdlog::error("Failed to retrieve tokens: {}", tokensResult.error().message());
    return false;
  }
  _parser.setTokens(std::move(tokensResult.value()));
  auto astResult = _parser.parse();
  if (!astResult) {
    spdlog::error("Failed to parse AST: {}", astResult.error().message());
    return false;
  }
  auto ast = std::move(astResult.value());
  _parser.printAST(ast);
  ast->normalizeTopLevelStatements();

  spdlog::debug("Generating initial MLIR");
  auto module = _mlirGen.generate(ast.get(), inputFile);
  spdlog::debug("Finished generating initial MLIR");
  spdlog::debug("Running pass manager");
  bool result = _passManager.run(module);
  if (!result) {
    spdlog::error("Failed to run pass manager!");
    return false;
  }
  spdlog::debug("Finished running pass manager");

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

  spdlog::debug("Finished translating MLIR to LLVM IR");
  if (!llvmModule) {
    spdlog::error("Failed to translate MLIR module to LLVM IR");
    return false;
  }

  spdlog::debug("Emitting object file");
  auto success = emitObjectFile(llvmModule.get(), "picceler.o");
  if (!success) {
    spdlog::error("Failed to emit an object file");
    return false;
  }

  spdlog::debug("Linking with LLD");
  success = linkWithLLD("picceler.o", "lib/libPiccelerRuntime.a", outputFile);
  if (!success) {
    spdlog::error("Failed to link an executable");
    return false;
  }

  return true;
}

bool Compiler::emitObjectFile(llvm::Module *llvmModule, const std::string &objFilename) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  std::string error;
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    spdlog::error("Target not found: {}", error);
    return false;
  }

  llvm::TargetOptions opt;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(llvm::Triple(targetTriple), "generic", "", opt,
                                  std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_)));

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(llvm::Triple(targetTriple));

  std::error_code ec;
  llvm::raw_fd_ostream dest(objFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    spdlog::error("Could not open file: {}", ec.message());
    return false;
  }

  llvm::legacy::PassManager pass;
  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
    spdlog::error("TargetMachine can't emit a file of this type");
    return false;
  }

  pass.run(*llvmModule);
  dest.flush();
  return true;
}

bool Compiler::linkWithLLD(const std::string &objFile, const std::string &runtimeLib, const std::string &outputExe) {
  std::string runtimeLibPath = runtimeLib;
  if (!std::filesystem::exists(objFile)) {
    spdlog::error("Object file not found for linking: {}", objFile);
    return false;
  }
  if (!std::filesystem::exists(runtimeLibPath)) {
    // Try common build output locations (e.g., build/lib/...) as a fallback when running from repo root
    const std::filesystem::path fallback = std::filesystem::path("build") / runtimeLibPath;
    if (std::filesystem::exists(fallback)) {
      spdlog::warn("Runtime library not found at '{}', using '{}' instead", runtimeLibPath, fallback.string());
      runtimeLibPath = fallback.string();
    } else {
      spdlog::error("Runtime library not found for linking: {}", runtimeLibPath);
      return false;
    }
  }

  std::vector<std::string> searchDirs = splitDelimited(PICCELER_OPENCV_LINK_DIRS, ';');
  for (const auto &dir : splitDelimited(PICCELER_CXX_IMPLICIT_LINK_DIRS, ';')) {
    appendUnique(searchDirs, dir);
  }
  for (const auto &defaultDir : kDefaultLibrarySearchDirs) {
    appendUnique(searchDirs, std::string(defaultDir));
  }
  for (const auto &gccDir : collectGccLibraryDirs()) {
    appendUnique(searchDirs, gccDir);
  }

  std::string openCvError;
  const auto openCvLibraries = collectOpenCvLibraries(searchDirs, openCvError);
  if (!openCvError.empty()) {
    spdlog::error("{}", openCvError);
    spdlog::error("Searched OpenCV library directories: {}", joinForLog(searchDirs));
    return false;
  }

  std::vector<std::string> startupSearchDirs = searchDirs;
  appendUnique(startupSearchDirs, "/usr/lib/x86_64-linux-gnu");
  appendUnique(startupSearchDirs, "/lib/x86_64-linux-gnu");

  // Prefer shared startup objects for PIE/dynamic linking when available
  std::string crt1Path;
  if (auto p = findStartupObject("Scrt1.o", startupSearchDirs))
    crt1Path = *p;
  else if (auto p = findStartupObject("crt1.o", startupSearchDirs))
    crt1Path = *p;

  std::string crtiPath;
  if (auto p = findStartupObject("crti.o", startupSearchDirs))
    crtiPath = *p;

  std::string crtbeginPath;
  if (auto p = findStartupObject("crtbeginS.o", startupSearchDirs))
    crtbeginPath = *p;
  else if (auto p = findStartupObject("crtbegin.o", startupSearchDirs))
    crtbeginPath = *p;

  std::string crtendPath;
  if (auto p = findStartupObject("crtendS.o", startupSearchDirs))
    crtendPath = *p;
  else if (auto p = findStartupObject("crtend.o", startupSearchDirs))
    crtendPath = *p;

  std::string crtnPath;
  if (auto p = findStartupObject("crtn.o", startupSearchDirs))
    crtnPath = *p;

  if (crt1Path.empty() || crtiPath.empty() || crtbeginPath.empty() || crtendPath.empty() || crtnPath.empty()) {
    spdlog::error("Missing system startup objects required by LLD linking");
    spdlog::error("Searched startup directories: {}", joinForLog(startupSearchDirs));
    return false;
  }

  std::vector<std::string> argsStorage = {"ld.lld", "-o", outputExe, "-pie", "-dynamic-linker", "/lib64/ld-linux-x86-64.so.2"};
  for (const auto &dir : searchDirs) {
    argsStorage.push_back("-L" + dir);
  }
  argsStorage.push_back(crt1Path);
  argsStorage.push_back(crtiPath);
  argsStorage.push_back(crtbeginPath);
  argsStorage.push_back(objFile);
  argsStorage.push_back(runtimeLibPath);
  argsStorage.insert(argsStorage.end(), openCvLibraries.begin(), openCvLibraries.end());

  const auto implicitArgs = collectImplicitLinkerArgs(searchDirs);
  // Filter out undesirable static libgcc.a that can break dynamic relocation ordering
  for (const auto &arg : implicitArgs) {
    if (arg.find("libgcc.a") != std::string::npos) {
      spdlog::info("Skipping implicit static lib: {}", arg);
      continue;
    }
    argsStorage.push_back(arg);
  }

  argsStorage.push_back("-lspdlog");
  argsStorage.push_back("-lfmt");
  argsStorage.push_back("-lpthread");
  argsStorage.push_back("-ldl");

  argsStorage.push_back("-lc");
  argsStorage.push_back(crtendPath);
  argsStorage.push_back(crtnPath);

  spdlog::debug("Linker args: {}", joinForLog(argsStorage));

  std::vector<const char *> argv;
  argv.reserve(argsStorage.size());
  for (const auto &arg : argsStorage) {
    argv.push_back(arg.c_str());
  }

  std::string linkStdout;
  std::string linkStderr;
  // Attempt external ld.lld invocation (avoids in-process LLD ABI/CLI clashes)
  int outPipe[2];
  int errPipe[2];
  if (pipe(outPipe) != 0 || pipe(errPipe) != 0) {
    spdlog::error("Failed to create pipes for linker I/O");
    return false;
  }

  pid_t pid = fork();
  if (pid == -1) {
    spdlog::error("fork() failed: {}", strerror(errno));
    return false;
  }

  if (pid == 0) {
    // Child: redirect stdout/stderr then exec ld.lld
    dup2(outPipe[1], STDOUT_FILENO);
    dup2(errPipe[1], STDERR_FILENO);
    close(outPipe[0]);
    close(outPipe[1]);
    close(errPipe[0]);
    close(errPipe[1]);

    std::vector<char *> execArgs;
    execArgs.reserve(argv.size() + 1);
    for (const char *a : argv)
      execArgs.push_back(const_cast<char *>(a));
    execArgs.push_back(nullptr);

    execvp("ld.lld", execArgs.data());
    // If exec fails
    _exit(127);
  }

  // Parent: read child's stdout/stderr
  close(outPipe[1]);
  close(errPipe[1]);

  // Read in separate buffers
  char buffer[4096];
  ssize_t n = 0;
  while ((n = read(outPipe[0], buffer, sizeof(buffer))) > 0) {
    linkStdout.append(buffer, buffer + n);
  }
  while ((n = read(errPipe[0], buffer, sizeof(buffer))) > 0) {
    linkStderr.append(buffer, buffer + n);
  }
  close(outPipe[0]);
  close(errPipe[0]);

  int status = 0;
  if (waitpid(pid, &status, 0) == -1) {
    spdlog::error("waitpid failed: {}", strerror(errno));
    return false;
  }
  const bool linked = (WIFEXITED(status) && WEXITSTATUS(status) == 0);

  if (!linkStdout.empty()) {
    spdlog::debug("LLD stdout:\n{}", linkStdout);
  }
  if (!linked) {
    spdlog::error("LLD linking failed for output '{}'", outputExe);
    if (!linkStderr.empty()) {
      spdlog::error("LLD diagnostics:\n{}", linkStderr);
    }
    return false;
  }

  spdlog::info("Successfully linked executable: {}", outputExe);
  if (!linkStderr.empty()) {
    spdlog::warn("LLD reported warnings:\n{}", linkStderr);
  }
  return true;
}

} // namespace picceler
