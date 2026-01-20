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

namespace picceler {

Compiler::Compiler()
    : _cliApp("picceler compiler"), _cliOptions(), _parser(), _context(initRegistry()), _mlirGen(&_context),
      _passManager(&_context) {
  spdlog::cfg::load_env_levels();
  _cliApp.add_option("input_file", _cliOptions.inputFile, "Input source file")->required()->check(CLI::ExistingFile);
  _cliApp.add_option("-o,--output", _cliOptions.outputFile, "Output executable file")->default_val("a.out");
  _context.loadDialect<PiccelerDialect>();
  _context.loadDialect<mlir::func::FuncDialect>();
  _context.loadDialect<mlir::arith::ArithDialect>();
  _context.loadDialect<mlir::affine::AffineDialect>();
  _context.loadDialect<mlir::memref::MemRefDialect>();
  _context.loadDialect<mlir::scf::SCFDialect>();
  _context.loadDialect<mlir::LLVM::LLVMDialect>();
  spdlog::debug("Initialized MLIR Dialects:");
  for (auto *dialect : _context.getLoadedDialects()) {
    spdlog::debug(" - {}", dialect->getNamespace());
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
  auto &cliOptions = getCliOptions();
  const auto &inputFile = _cliOptions.inputFile;
  const auto &outputFile = _cliOptions.outputFile;

  _parser.setSource(inputFile);
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
  _parser.setSource(inputFile);
  auto ast = _parser.parse();
  if (!ast) {
    spdlog::error("{}", ast.error().message());
    return false;
  }
  _parser.printAST(ast.value());

  spdlog::info("Generating initial MLIR");
  auto module = _mlirGen.generate(ast.value().get());
  spdlog::info("Finished generating initial MLIR");
  module->dump();
  spdlog::info("Running pass manager");
  _passManager.run(module);
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

  success = linkWithLLD("picceler.o", "lib/libpicceler_runtime.a", outputFile);
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

bool Compiler::linkWithLLD(const std::string &objFile, const std::string &runtimeLib, const std::string &outputExe) {
  // Build the link command using clang++ with -no-pie to avoid PIE relocations
  std::string cmd =
      "clang++ " + objFile + " " + runtimeLib + " -o " + outputExe + " -no-pie $(pkg-config --libs opencv4)";
  spdlog::debug("Linking with command: {}", cmd);

  int ret = std::system(cmd.c_str());
  if (ret == 0) {
    spdlog::info("Successfully linked executable: {}", outputExe);
    return true;
  } else {
    spdlog::error("Linking failed with exit code: {}", ret);
    return false;
  }
}

} // namespace picceler