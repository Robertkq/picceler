#include "parser.h"
#include "lexer.h"
#include <gtest/gtest.h>

namespace picceler {

class ParserTest : public ::testing::Test {
protected:
  // Helper to lex source text into tokens
  std::vector<Token> tokenize(std::string_view source) {
    Lexer lexer;
    lexer.setSourceString(source);
    auto result = lexer.getTokens();
    if (!result) {
      ADD_FAILURE() << "Tokenization failed unexpectedly: " << result.error().message();
      return {};
    }
    return std::move(*result);
  }

  // Helper to run Lexer -> Parser pipeline successfully
  std::unique_ptr<ModuleNode> parseSuccessfully(std::string_view source) {
    auto tokens = tokenize(source);
    Parser parser(std::move(tokens));

    auto result = parser.parse();
    if (!result) {
      ADD_FAILURE() << "Parsing failed unexpectedly: " << result.error().message();
      return nullptr;
    }
    return std::move(*result);
  }

  // Helper to assert that parsing fails
  void assertParseFails(std::string_view source) {
    auto tokens = tokenize(source);
    Parser parser(std::move(tokens));
    auto result = parser.parse();
    EXPECT_FALSE(result.has_value()) << "Expected parse failure for input: " << source;
  }
};

// --- Downcast Helper to remove dynamic_cast boilerplate ---

template <typename TargetNode, typename BaseNode> const TargetNode *as(const BaseNode *node) {
  const auto *casted = dynamic_cast<const TargetNode *>(node);
  EXPECT_NE(casted, nullptr) << "Failed to downcast AST node!";
  return casted;
}

// --- Tests ---

TEST_F(ParserTest, EmptyInput) {
  auto ast = parseSuccessfully("");
  ASSERT_NE(ast, nullptr);
  EXPECT_EQ(ast->statements().size(), 0);
}

TEST_F(ParserTest, BadKernelSyntaxFails) {
  assertParseFails("k = [[1 2],[3,4]]");
  assertParseFails("k = [[1,2],[3,4]");
}

TEST_F(ParserTest, LoadImageStatement) {
  auto ast = parseSuccessfully(R"(img = load_image("cat.jpg"))");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements().size(), 1);

  const auto *assign = as<AssignmentNode>(ast->statements()[0]);
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->lhs()->name(), "img");

  const auto *call = as<CallNode>(assign->rhs());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->callee(), "load_image");
  ASSERT_EQ(call->arguments().size(), 1);

  const auto *strArg = as<StringNode>(call->arguments()[0]);
  ASSERT_NE(strArg, nullptr);
  EXPECT_EQ(strArg->value(), "cat.jpg");
}

TEST_F(ParserTest, RotateNegativeAngleParses) {
  auto ast = parseSuccessfully("img = rotate(input, -90)");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements().size(), 1);

  const auto *assign = as<AssignmentNode>(ast->statements()[0]);
  ASSERT_NE(assign, nullptr);

  const auto *call = as<CallNode>(assign->rhs());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->callee(), "rotate");
  ASSERT_EQ(call->arguments().size(), 2);

  const auto *angleNode = as<NumberNode>(call->arguments()[1]);
  ASSERT_NE(angleNode, nullptr);
  EXPECT_EQ(angleNode->value(), -90);
}

} // namespace picceler