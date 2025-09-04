#include "lexer.h"
#include <gtest/gtest.h>

class LexerTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  picceler::Lexer lexer;
};

TEST_F(LexerTest, EmptyInput) {
  lexer.setSource("tests/data/empty.pic");
  auto tokens = lexer.tokenizeAll();
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0]._type, picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, LoadImageStatement) {
  lexer.setSource("tests/data/load_image.pic");
  // file contents:
  // img = load_image("cat.jpg")
  auto tokens = lexer.tokenizeAll();

  ASSERT_GE(tokens.size(), 7); // img, =, load_image, (, "cat.jpg", ), EOF

  EXPECT_EQ(tokens[0]._type, picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(tokens[0]._value, "img");

  EXPECT_EQ(tokens[1]._type, picceler::Token::Type::SYMBOL);
  EXPECT_EQ(tokens[1]._value, "=");

  EXPECT_EQ(tokens[2]._type, picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(tokens[2]._value, "load_image");

  EXPECT_EQ(tokens[3]._type, picceler::Token::Type::SYMBOL);
  EXPECT_EQ(tokens[3]._value, "(");

  EXPECT_EQ(tokens[4]._type, picceler::Token::Type::STRING);
  EXPECT_EQ(tokens[4]._value, "cat.jpg");

  EXPECT_EQ(tokens[5]._type, picceler::Token::Type::SYMBOL);
  EXPECT_EQ(tokens[5]._value, ")");

  EXPECT_EQ(tokens.back()._type, picceler::Token::Type::EOF_TOKEN);
}
