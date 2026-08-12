#include "compiler.h"

#include <cctype>
#include <filesystem>
#include <string_view>
#include <vector>

#include "mlir/Dialect/Math/IR/Math.h"
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

struct ProcessResult {
  int exitCode = -1;
  std::string stdOut;
  std::string stdErr;
};

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

ProcessResult runCommand(const std::vector<std::string> &args) {
  ProcessResult result;
  if (args.empty()) {
    return result;
  }

  int outPipe[2];
  int errPipe[2];
  if (pipe(outPipe) != 0) {
    spdlog::error("pipe() failed for stdout: {}", strerror(errno));
    return result;
  }
  if (pipe(errPipe) != 0) {
    spdlog::error("pipe() failed for stderr: {}", strerror(errno));
    close(outPipe[0]);
    close(outPipe[1]);
    return result;
  }

  pid_t pid = fork();
  if (pid == -1) {
    spdlog::error("fork() failed: {}", strerror(errno));
    close(outPipe[0]);
    close(outPipe[1]);
    close(errPipe[0]);
    close(errPipe[1]);
    return result;
  }

  if (pid == 0) {
    dup2(outPipe[1], STDOUT_FILENO);
    dup2(errPipe[1], STDERR_FILENO);

    close(outPipe[0]);
    close(outPipe[1]);
    close(errPipe[0]);
    close(errPipe[1]);

    std::vector<char *> execArgs;
    execArgs.reserve(args.size() + 1);
    for (const auto &arg : args) {
      execArgs.push_back(const_cast<char *>(arg.c_str()));
    }
    execArgs.push_back(nullptr);

    execvp(execArgs[0], execArgs.data());
    _exit(127);
  }

  close(outPipe[1]);
  close(errPipe[1]);

  char buffer[4096];
  ssize_t bytesRead = 0;
  while ((bytesRead = read(outPipe[0], buffer, sizeof(buffer))) > 0) {
    result.stdOut.append(buffer, bytesRead);
  }
  while ((bytesRead = read(errPipe[0], buffer, sizeof(buffer))) > 0) {
    result.stdErr.append(buffer, bytesRead);
  }

  close(outPipe[0]);
  close(errPipe[0]);

  int status = 0;
  if (waitpid(pid, &status, 0) != -1 && WIFEXITED(status)) {
    result.exitCode = WEXITSTATUS(status);
  }

  return result;
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
  registry.insert<mlir::math::MathDialect>();
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

  spdlog::debug("Linking with Clang");
  success = linkWithClang("picceler.o", "lib/libPiccelerRuntime.a", outputFile);
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
                                  std::optional<llvm::Reloc::Model>(llvm::Reloc::Static)));

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

bool Compiler::linkWithClang(const std::string &objFile, const std::string &runtimeLib, const std::string &outputExe) {
  if (!std::filesystem::exists(objFile)) {
    spdlog::error("Object file not found for linking: {}", objFile);
    return false;
  }

  std::string runtimeLibPath = runtimeLib;
  if (!std::filesystem::exists(runtimeLibPath)) {
    const std::filesystem::path fallback = std::filesystem::path("build") / runtimeLibPath;
    if (std::filesystem::exists(fallback)) {
      spdlog::warn("Runtime library not found at '{}', using '{}' instead", runtimeLibPath, fallback.string());
      runtimeLibPath = fallback.string();
    } else {
      spdlog::error("Runtime library not found for linking: {}", runtimeLibPath);
      return false;
    }
  }

  spdlog::info("Querying pkg-config for opencv4...");
  ProcessResult pkgResult = runCommand({"pkg-config", "--libs", "opencv4"});
  if (pkgResult.exitCode != 0) {
    spdlog::error("pkg-config failed:");
    if (!pkgResult.stdErr.empty()) {
      spdlog::error("pkg-config diagnostics:\n{}", pkgResult.stdErr);
    }
    return false;
  }

  std::vector<std::string> clangArgs = {"clang++", objFile, runtimeLibPath, "-o", outputExe, "-no-pie"};
  for (const auto &token : splitWhitespace(pkgResult.stdOut)) {
    clangArgs.push_back(token);
  }
  clangArgs.push_back("-lspdlog");
  clangArgs.push_back("-lfmt");
  clangArgs.push_back("-lpthread");

  spdlog::info("Linking executable: {}", outputExe);
  ProcessResult clangResult = runCommand(clangArgs);
  if (clangResult.exitCode != 0) {
    spdlog::error("clang++ linking failed!");
    if (!clangResult.stdErr.empty()) {
      spdlog::error("Linker output:\n{}", clangResult.stdErr);
    }
    return false;
  }

  spdlog::info("Successfully linked executable: {}", outputExe);
  if (!clangResult.stdErr.empty()) {
    spdlog::warn("Linker warnings:\n{}", clangResult.stdErr);
  }

  return true;
}

} // namespace picceler
