#include "parser.h"
#include <gtest/gtest.h>

class ParserTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  picceler::Parser parser;
};

TEST_F(ParserTest, BadKernelMissingCommaFails) {
  parser.setSource("data/bad_kernel_missing_comma.pic");

  auto astRes = parser.parse();
  if (!astRes)
    FAIL() << astRes.error().message();
  EXPECT_FALSE(astRes.has_value());
}

TEST_F(ParserTest, BadKernelMissingBracketFails) {
  parser.setSource("data/bad_kernel_missing_bracket.pic");

  auto astRes = parser.parse();
  if (!astRes)
    FAIL() << astRes.error().message();
  EXPECT_FALSE(astRes.has_value());
}

TEST_F(ParserTest, BadKernelBadNumberFails) {
  parser.setSource("data/bad_kernel_bad_number.pic");
  auto astRes = parser.parse();
  if (!astRes)
    FAIL() << astRes.error().message();
  EXPECT_FALSE(astRes.has_value());
}

TEST_F(ParserTest, EmptyInput) {
  parser.setSource("data/empty.pic");

  auto astRes = parser.parse();
  if (!astRes)
    FAIL() << astRes.error().message();
  auto ast = std::move(astRes.value());

  ASSERT_NE(ast, nullptr);
  EXPECT_EQ(ast->statements.size(), 0);
}

TEST_F(ParserTest, LoadImageStatement) {
  parser.setSource("tests/data/load_image.pic");

  // file contents:
  // img = load_image("cat.jpg")
  auto astRes = parser.parse();
  if (!astRes)
    FAIL() << astRes.error().message();
  auto ast = std::move(astRes.value());

  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements.size(), 1);
  auto assignment = dynamic_cast<picceler::AssignmentNode *>(ast->statements[0].get());
  ASSERT_NE(assignment, nullptr);
  EXPECT_EQ(assignment->lhs->name, "img");
  auto call = dynamic_cast<picceler::CallNode *>(assignment->rhs.get());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->callee, "load_image");
  ASSERT_EQ(call->arguments.size(), 1);
  auto strArg = dynamic_cast<picceler::StringNode *>(call->arguments[0].get());
  ASSERT_NE(strArg, nullptr);
  EXPECT_EQ(strArg->value, "cat.jpg");
}