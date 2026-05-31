#include "compiler.h"

#include <cstdlib>

#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

namespace picceler {

static llvm::cl::opt<std::string> InputFile(llvm::cl::Positional,
    llvm::cl::desc("<input file>"), llvm::cl::Required);

static llvm::cl::opt<std::string> OutputFile("o",
    llvm::cl::desc("Output executable file"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init("a.out"));

Compiler::Compiler()
    : _cliOptions(), _parser(), _context(nullptr), _mlirGen(nullptr), _passManager(nullptr) {
  spdlog::cfg::load_env_levels();
}

bool Compiler::parseArgs(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Picceler compiler\n");
  _cliOptions.inputFile = InputFile;
  _cliOptions.outputFile = OutputFile;

  // Validate input file exists
  if (!llvm::sys::fs::exists(_cliOptions.inputFile)) {
    spdlog::error("Input file does not exist: {}", _cliOptions.inputFile);
    return false;
  }

  return true;
}

mlir::MLIRContext &Compiler::getContext() {
  // Lazy initialization: only create MLIR context after CLI parsing is complete.
  // This avoids conflicts between CLI11 and LLVM's global command-line infrastructure.
  if (!_context) {
    auto registry = initRegistry();
    _context = std::make_unique<mlir::MLIRContext>(registry);
    _mlirGen = std::make_unique<MLIRGen>(_context.get());
    _passManager = std::make_unique<IRPassManager>(_context.get());

    _context->loadAllAvailableDialects();
    spdlog::debug("Initialized MLIR Dialects:");
    for (auto *dialect : _context->getLoadedDialects()) {
      spdlog::debug(" - {}", dialect->getNamespace());
    }
  }
  return *_context;
}

static std::string findRuntimeLibrary();

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
  // Ensure MLIR context is initialized before proceeding
  getContext();

  auto &cliOptions = getCliOptions();
  const auto &inputFile = _cliOptions.inputFile;
  const auto &outputFile = _cliOptions.outputFile;

  auto sourceResult = _parser.setSource(inputFile);
  if (!sourceResult) {
    spdlog::error("Failed to set source file: {}", sourceResult.error().message());
    return false;
  }
  spdlog::info("Tokenizing source file: {}", inputFile);
  auto tokens = _parser.getTokens();
  if (!tokens) {
    spdlog::error("Failed to tokenize source file: {}", tokens.error().message());
    return false;
  }
  size_t index = 0;
  for (const auto &token : tokens.value()) {
    spdlog::debug("Token[{}]: {}", index++, token.toString());
  }
  spdlog::info("Finished tokenizing source file");
  spdlog::info("Resetting lexer");
  auto resetResult = _parser.setSource(inputFile);
  if (!resetResult) {
    spdlog::error("Failed to reset source file: {}", resetResult.error().message());
    return false;
  }
  auto ast = _parser.parse();
  if (!ast) {
    spdlog::error("{}", ast.error().message());
    return false;
  }
  _parser.printAST(ast.value());

  spdlog::info("Generating initial MLIR");
  auto module = _mlirGen->generate(ast.value().get());
  spdlog::info("Finished generating initial MLIR");
  module->dump();
  spdlog::info("Running pass manager");
  _passManager->run(module);
  spdlog::info("Finished running pass manager");

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

  if (!llvmModule) {
    spdlog::error("Failed to translate MLIR module to LLVM IR");
    return false;
  }

  auto success = emitObjectFile(llvmModule.get(), "picceler.o");
  if (!success) {
    spdlog::error("Failed to emit an object file");
    return false;
  }

  const std::string runtimeLib = findRuntimeLibrary();
  if (runtimeLib.empty()) {
    spdlog::error("Compiler Error: Picceler runtime library could not be found. Expected one of:");
    spdlog::error("  build/lib/libPiccelerRuntime.a");
    spdlog::error("  lib/libPiccelerRuntime.a");
    spdlog::error("  ../build/lib/libPiccelerRuntime.a");
    spdlog::error("  ../lib/libPiccelerRuntime.a");
    return false;
  }

  success = linkWithLLD("picceler.o", runtimeLib, outputFile);
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
      target->createTargetMachine(llvm::Triple(targetTriple), "generic", "", opt, std::nullopt));

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(llvm::Triple(targetTriple));

  std::error_code EC;
  llvm::raw_fd_ostream dest(objFilename, EC, llvm::sys::fs::OF_None);
  if (EC) {
    spdlog::error("Could not open file: {}", EC.message());
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

std::string findGccLibDir() {
  std::error_code EC;
  std::string basePath = "/usr/lib/gcc/x86_64-pc-linux-gnu";
  if (!llvm::sys::fs::exists(basePath)) return "";

  std::string latestVersion = "";
  for (llvm::sys::fs::directory_iterator dir(basePath, EC), end; !EC && dir != end; dir.increment(EC)) {
      llvm::StringRef filename = llvm::sys::path::filename(dir->path());
      if (filename.str() > latestVersion) {
          latestVersion = filename.str();
      }
  }
  return latestVersion.empty() ? "" : basePath + "/" + latestVersion;
}

static std::string findRuntimeLibrary() {
  llvm::SmallString<256> cwd;
  if (auto ec = llvm::sys::fs::current_path(cwd)) {
    spdlog::warn("Unable to determine current working directory: {}", ec.message());
    return "";
  }

  std::vector<llvm::SmallString<256>> candidates;
  candidates.emplace_back(cwd);
  llvm::sys::path::append(candidates.back(), "build");
  llvm::sys::path::append(candidates.back(), "lib");
  llvm::sys::path::append(candidates.back(), "libPiccelerRuntime.a");

  candidates.emplace_back(cwd);
  llvm::sys::path::append(candidates.back(), "lib");
  llvm::sys::path::append(candidates.back(), "libPiccelerRuntime.a");

  candidates.emplace_back(cwd);
  llvm::sys::path::append(candidates.back(), "..", "build");
  llvm::sys::path::append(candidates.back(), "lib");
  llvm::sys::path::append(candidates.back(), "libPiccelerRuntime.a");

  candidates.emplace_back(cwd);
  llvm::sys::path::append(candidates.back(), "..", "lib");
  llvm::sys::path::append(candidates.back(), "libPiccelerRuntime.a");

  candidates.emplace_back(cwd);
  llvm::sys::path::append(candidates.back(), "..", "..", "build");
  llvm::sys::path::append(candidates.back(), "lib");
  llvm::sys::path::append(candidates.back(), "libPiccelerRuntime.a");

  for (const auto &candidate : candidates) {
    if (llvm::sys::fs::exists(candidate)) {
      return candidate.str().str();
    }
  }

  return "";
}

bool Compiler::linkWithLLD(const std::string &objFile, const std::string &runtimeLib, const std::string &outputExe) {
  spdlog::info("Starting linking for {}", outputExe);

  if (!llvm::sys::fs::exists(objFile)) {
    spdlog::error("Compiler Error: Generated object file not found at {}", objFile);
    return false;
  }

  if (!llvm::sys::fs::exists(runtimeLib)) {
    spdlog::error("Compiler Error: Picceler runtime library missing at {}", runtimeLib);
    return false;
  }

  std::string gccLibDir = findGccLibDir();
  if (gccLibDir.empty()) {
    spdlog::warn("Could not dynamically find GCC lib directory. C++ global constructors/destructors might fail.");
  }

  std::vector<std::string> argsStr = {
    "ld.lld",
    "--dynamic-linker", "/lib64/ld-linux-x86-64.so.2",
    "-o", outputExe,

    "/usr/lib/Scrt1.o",
    "/usr/lib/crti.o",
    gccLibDir.empty() ? "" : gccLibDir + "/crtbeginS.o",

    objFile,
    runtimeLib,

    "-L/usr/lib",
    "-L/usr/local/lib",
    "-lopencv_core",
    "-lopencv_imgproc",
    "-lopencv_highgui",
    "-lopencv_imgcodecs",
    "-lfmt",

    "-lstdc++",
    "-lm",
    gccLibDir.empty() ? "" : "-L" + gccLibDir,
    "-lgcc_s",
    "-lgcc",
    "-lc",

    gccLibDir.empty() ? "" : gccLibDir + "/crtendS.o",
    "/usr/lib/crtn.o"
  };

  // Find ld.lld in the environment rather than hardcoding its path.
  auto lldPath = llvm::sys::findProgramByName("ld.lld");
  if (!lldPath) {
    spdlog::error("Could not find ld.lld in PATH. Install LLD or add it to PATH.");
    return false;
  }

  // Filter out empty strings and convert to StringRef
  std::vector<llvm::StringRef> args;
  for (const auto& arg : argsStr) {
      if (!arg.empty()) {
          args.push_back(llvm::StringRef(arg));
      }
  }

  spdlog::debug("Invoking linker with {} arguments", args.size());

  // Use LLVM's ExecuteAndWait to invoke the system ld.lld linker.
  // This avoids std::system() and keeps the linkage invocation inside LLVM tooling.
  int result = llvm::sys::ExecuteAndWait(
    *lldPath,
    args,
    std::nullopt,  // Use current environment
    {},            // No redirects; inherit stdout/stderr
    0,             // No timeout
    0              // No memory limit
  );

  if (result == 0) {
    spdlog::info("Successfully linked executable: {}", outputExe);
    return true;
  } else {
    spdlog::error("Linker phase failed with exit code: {}", result);

    if (result == 127) {
      spdlog::error("Diagnostic: ld.lld not found at /usr/bin/ld.lld. Check your LLD installation.");
    } else {
      spdlog::error("Diagnostic: Check linker error messages above for details.");
    }

    return false;
  }
}

} // namespace picceler
