#include "app.h"

namespace picceler {

App::App() : _cliApp("picceler compiler"), _inputFile(), _parser() {
  _cliApp.add_option("-i,--input", _inputFile, "Input source file")->required();
}

void App::test() {
  _parser.setSource(_inputFile);
  auto tokens = _parser.getTokens();
  for (const auto &token : tokens) {
    std::cout << token << "\n";
  }
}

} // namespace picceler