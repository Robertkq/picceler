#include "parser.h"
#include "ast.h"

#include <gtest/gtest.h>

#include <string>
#include <memory>

using namespace picceler;

struct ASTTestCase {
  std::string name;
  std::string input;
  std::string expectedAST;
};

class ASTParameterizedTest : public ::testing::TestWithParam<ASTTestCase> {
protected:
  std::unique_ptr<ModuleNode> parseOrFail(const std::string &input) {
    _parser.setSourceString(input);
    auto astResult = _parser.parse();
    EXPECT_TRUE(astResult.has_value()) << "Parsing failed with message: " << astResult.error().message();
    EXPECT_NE(astResult.value(), nullptr);
    return std::move(astResult.value());
  }

protected:
  picceler::Parser _parser;
};

TEST_P(ASTParameterizedTest, MatchesExpectedStringRepresentation) {
  const ASTTestCase &testCase = GetParam();
  auto ast = parseOrFail(testCase.input);
  ASSERT_NE(ast, nullptr);
  std::string actualAST = ast->toString();
  EXPECT_EQ(actualAST, testCase.expectedAST);
}

using namespace ::testing;

INSTANTIATE_TEST_SUITE_P(ASTTests, ASTParameterizedTest, Values(ASTTestCase{"EmptyModule", "", "Module: 0 statements"}),
                         [](const TestParamInfo<ASTTestCase> &info) { return info.param.name; });