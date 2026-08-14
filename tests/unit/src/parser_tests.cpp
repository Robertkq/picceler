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

TEST_F(ParserTest, NestedFunctionCalls) {
  auto ast = parseSuccessfully(R"(out = blur(load_image("cat.jpg"), 5))");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements().size(), 1);

  const auto *assign = as<AssignmentNode>(ast->statements()[0]);
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->lhs()->name(), "out");

  const auto *outerCall = as<CallNode>(assign->rhs());
  ASSERT_NE(outerCall, nullptr);
  EXPECT_EQ(outerCall->callee(), "blur");
  ASSERT_EQ(outerCall->arguments().size(), 2);

  const auto *innerCall = as<CallNode>(outerCall->arguments()[0]);
  ASSERT_NE(innerCall, nullptr);
  EXPECT_EQ(innerCall->callee(), "load_image");
  ASSERT_EQ(innerCall->arguments().size(), 1);

  const auto *strArg = as<StringNode>(innerCall->arguments()[0]);
  ASSERT_NE(strArg, nullptr);
  EXPECT_EQ(strArg->value(), "cat.jpg");
}

TEST_F(ParserTest, MultipleStatements) {
  auto ast = parseSuccessfully(R"(
      a = 1
      b = a
  )");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements().size(), 2);

  const auto *stmt1 = as<AssignmentNode>(ast->statements()[0]);
  ASSERT_NE(stmt1, nullptr);
  EXPECT_EQ(stmt1->lhs()->name(), "a");

  const auto *stmt2 = as<AssignmentNode>(ast->statements()[1]);
  ASSERT_NE(stmt2, nullptr);
  EXPECT_EQ(stmt2->lhs()->name(), "b");
}

TEST_F(ParserTest, UnclosedParenFails) { assertParseFails(R"(img = load_image("cat.jpg")"); }

TEST_F(ParserTest, ArithmeticPrecedenceAndBinaryOps) {
  // 2 + 3 * 4 should parse as 2 + (3 * 4) because multiplication has higher precedence
  auto ast = parseSuccessfully("res = 2 + 3 * 4");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements().size(), 1);

  const auto *assign = as<AssignmentNode>(ast->statements()[0]);
  ASSERT_NE(assign, nullptr);

  const auto *addNode = as<BinaryOpNode>(assign->rhs());
  ASSERT_NE(addNode, nullptr);
  EXPECT_EQ(addNode->op(), "+");

  // LHS of addition should be 2
  const auto *leftNum = as<NumberNode>(addNode->lhs());
  ASSERT_NE(leftNum, nullptr);
  EXPECT_EQ(leftNum->value(), 2);

  // RHS of addition should be a multiplication node (3 * 4)
  const auto *mulNode = as<BinaryOpNode>(addNode->rhs());
  ASSERT_NE(mulNode, nullptr);
  EXPECT_EQ(mulNode->op(), "*");

  const auto *mulLeft = as<NumberNode>(mulNode->lhs());
  EXPECT_EQ(mulLeft->value(), 3);

  const auto *mulRight = as<NumberNode>(mulNode->rhs());
  EXPECT_EQ(mulRight->value(), 4);
}

TEST_F(ParserTest, ParenthesesOverridePrecedence) {
  // (2 + 3) * 4 forces addition to happen first
  auto ast = parseSuccessfully("res = (2 + 3) * 4");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements().size(), 1);

  const auto *assign = as<AssignmentNode>(ast->statements()[0]);
  const auto *mulNode = as<BinaryOpNode>(assign->rhs());
  ASSERT_NE(mulNode, nullptr);
  EXPECT_EQ(mulNode->op(), "*");

  // LHS of multiplication should be the addition node (2 + 3)
  const auto *addNode = as<BinaryOpNode>(mulNode->lhs());
  ASSERT_NE(addNode, nullptr);
  EXPECT_EQ(addNode->op(), "+");
}

TEST_F(ParserTest, RelationalComparisonExpression) {
  // Testing a complete user scenario: var * car <= 50
  auto ast = parseSuccessfully(R"(
    var = -1 + 2 * 3
    car = 10
    if (var * car <= 50) {
      print("x is greater than 2 \n")
    }
  )");

  ASSERT_NE(ast, nullptr);
  // Expecting 3 top-level statements: assignment, assignment, if-statement
  ASSERT_EQ(ast->statements().size(), 3);
}

TEST_F(ParserTest, SqrtFunctionParses) {
  auto ast = parseSuccessfully("res = sqrt(16.0)");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements().size(), 1);

  const auto *assign = as<AssignmentNode>(ast->statements()[0]);
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->lhs()->name(), "res");

  const auto *call = as<CallNode>(assign->rhs());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->callee(), "sqrt");
  ASSERT_EQ(call->arguments().size(), 1);

  const auto *arg = as<NumberNode>(call->arguments()[0]);
  ASSERT_NE(arg, nullptr);
  EXPECT_EQ(arg->value(), 16.0);
}

TEST_F(ParserTest, PowFunctionParses) {
  auto ast = parseSuccessfully("res = pow(2.0, 3.0)");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements().size(), 1);

  const auto *assign = as<AssignmentNode>(ast->statements()[0]);
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->lhs()->name(), "res");

  const auto *call = as<CallNode>(assign->rhs());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->callee(), "pow");
  ASSERT_EQ(call->arguments().size(), 2);

  const auto *base = as<NumberNode>(call->arguments()[0]);
  ASSERT_NE(base, nullptr);
  EXPECT_EQ(base->value(), 2.0);

  const auto *exp = as<NumberNode>(call->arguments()[1]);
  ASSERT_NE(exp, nullptr);
  EXPECT_EQ(exp->value(), 3.0);
}

TEST_F(ParserTest, ForLoopParsesSuccessfully) {
  auto ast = parseSuccessfully(R"(
    for (i = 1 .. 10) {
      a = i
    }
  )");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements().size(), 1);

  // Pass directly without .get()
  const auto *forNode = as<ForNode>(ast->statements()[0]);
  ASSERT_NE(forNode, nullptr);
  EXPECT_EQ(forNode->varName(), "i");

  // Check that lower bound is 1 and upper bound is 10
  const auto *lb = as<NumberNode>(forNode->lowerBound());
  ASSERT_NE(lb, nullptr);
  EXPECT_EQ(lb->value(), 1.0);

  const auto *ub = as<NumberNode>(forNode->upperBound());
  ASSERT_NE(ub, nullptr);
  EXPECT_EQ(ub->value(), 10.0);

  // Check body statement count
  ASSERT_EQ(forNode->body().size(), 1);

  // Pass directly without .get()
  const auto *assign = as<AssignmentNode>(forNode->body()[0].get());
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->lhs()->name(), "a");
}

TEST_F(ParserTest, ForLoopWithStepParsesSuccessfully) {
  auto ast = parseSuccessfully(R"(
    for (j = 0 .. 100 step 5) {
      print("j: {}", j)
    }
  )");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements().size(), 1);

  const auto *forNode = as<ForNode>(ast->statements()[0]);
  ASSERT_NE(forNode, nullptr);
  EXPECT_EQ(forNode->varName(), "j");

  // Check step is present and evaluates to 5
  const auto *stepNum = as<NumberNode>(forNode->step());
  ASSERT_NE(stepNum, nullptr);
  EXPECT_EQ(stepNum->value(), 5.0);

  // Check body statement count
  ASSERT_EQ(forNode->body().size(), 1);
}

TEST_F(ParserTest, InvalidForLoopFails) {
  // Missing '..' operator
  assertParseFails(R"(
    for (i = 1 10) {
      a = i
    }
  )");

  // Missing closing parenthesis
  assertParseFails(R"(
    for (i = 1 .. 10 {
      a = i
    }
  )");
}

} // namespace picceler
