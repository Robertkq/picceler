#include "parser.h"
#include <gtest/gtest.h>

class ParserTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  picceler::Parser parser;
};

TEST_F(ParserTest, EmptyInput) {
  parser.setSource("tests/data/empty.pic");
  auto ast = parser.parse();
  ASSERT_NE(ast, nullptr);
  EXPECT_EQ(ast->statements.size(), 0);
}

TEST_F(ParserTest, LoadImageStatement) {
  parser.setSource("tests/data/load_image.pic");
  // file contents:
  // img = load_image("cat.jpg")
  auto ast = parser.parse();
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->statements.size(), 1);
  auto assignment =
      dynamic_cast<picceler::AssignmentNode *>(ast->statements[0].get());
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