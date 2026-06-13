#include "lexer.h"
#include <gtest/gtest.h>

class LexerTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  picceler::Lexer lexer;
};

TEST_F(LexerTest, EmptyInput) {
  auto res = lexer.setSource("data/empty.pic");
  if (!res)
    FAIL() << res.error().message();

  auto tokensRes = lexer.tokenizeAll();
  if (!tokensRes)
    FAIL() << tokensRes.error().message();
  auto tokens = tokensRes.value();

  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, LoadImageStatement) {
  auto res = lexer.setSource("data/load_image.pic");
  if (!res)
    FAIL() << res.error().message();

  // file contents:
  // img = load_image("cat.jpg")
  auto tokensRes = lexer.tokenizeAll();
  if (!tokensRes)
    FAIL() << tokensRes.error().message();
  auto tokens = tokensRes.value();

  ASSERT_GE(tokens.size(), 7); // img, =, load_image, (, "cat.jpg", ), EOF

  EXPECT_EQ(tokens[0].type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(tokens[0].value(), "img");

  EXPECT_EQ(tokens[1].type(), picceler::Token::Type::ASSIGN);
  EXPECT_EQ(tokens[1].value(), "=");

  EXPECT_EQ(tokens[2].type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(tokens[2].value(), "load_image");

  EXPECT_EQ(tokens[3].type(), picceler::Token::Type::L_PAREN);
  EXPECT_EQ(tokens[3].value(), "(");

  EXPECT_EQ(tokens[4].type(), picceler::Token::Type::STRING);
  EXPECT_EQ(tokens[4].value(), "cat.jpg");

  EXPECT_EQ(tokens[5].type(), picceler::Token::Type::R_PAREN);
  EXPECT_EQ(tokens[5].value(), ")");

  EXPECT_EQ(tokens.back().type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, NumbersParsing) {
  auto res = lexer.setSource("data/numbers.pic");
  if (!res)
    FAIL() << res.error().message();

  auto tokensRes = lexer.tokenizeAll();
  if (!tokensRes)
    FAIL() << tokensRes.error().message();
  auto tokens = tokensRes.value();

  std::vector<std::string> numbers;
  for (const auto &t : tokens) {
    if (t.type() == picceler::Token::Type::NUMBER)
      numbers.push_back(t.value());
  }

  ASSERT_EQ(numbers.size(), 3);
  EXPECT_EQ(numbers[0], "123");
  EXPECT_EQ(numbers[1], "-3.14");
  EXPECT_EQ(numbers[2], "0.5");
}

TEST_F(LexerTest, KernelTokenSequence) {
  auto res = lexer.setSource("data/kernel.pic");
  if (!res)
    FAIL() << res.error().message();

  auto tokensRes = lexer.tokenizeAll();
  if (!tokensRes)
    FAIL() << tokensRes.error().message();
  auto tokens = tokensRes.value();

  // Expect sequence: IDENTIFIER, '=', '[', '[', NUMBER(1), ',', NUMBER(2), ']', ',', '[', NUMBER(3), ',', NUMBER(4),
  // ']', ']', EOF
  std::vector<std::string> seq;
  for (const auto &t : tokens) {
    if (t.type() == picceler::Token::Type::L_BRACKET)
      seq.push_back(t.value());
    else if (t.type() == picceler::Token::Type::R_BRACKET)
      seq.push_back(t.value());
    else if (t.type() == picceler::Token::Type::COMMA)
      seq.push_back(t.value());
    else if (t.type() == picceler::Token::Type::NUMBER)
      seq.push_back(t.value());
    else if (t.type() == picceler::Token::Type::IDENTIFIER)
      seq.push_back(t.value());
    else if (t.type() == picceler::Token::Type::ASSIGN)
      seq.push_back(t.value());
  }

  std::vector<std::string> expected = {"k", "=", "[", "[", "1", ",", "2", "]", ",", "[", "3", ",", "4", "]", "]"};
  // the tokens vector also includes EOF at the end; compare prefix
  ASSERT_GE(seq.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(seq[i], expected[i]);
  }
}

TEST_F(LexerTest, IdentifiersAndUnderscores) {
  auto res = lexer.setSource("data/identifiers.pic");
  if (!res)
    FAIL() << res.error().message();

  auto tokensRes = lexer.tokenizeAll();
  if (!tokensRes)
    FAIL() << tokensRes.error().message();
  auto tokens = tokensRes.value();

  // Check presence of identifier names
  bool foundMyVar = false;
  bool foundSomeFunction = false;
  bool foundPrivate = false;
  for (const auto &t : tokens) {
    if (t.type() == picceler::Token::Type::IDENTIFIER) {
      if (t.value() == "my_var")
        foundMyVar = true;
      if (t.value() == "some_function")
        foundSomeFunction = true;
      if (t.value() == "_private123")
        foundPrivate = true;
    }
  }
  EXPECT_TRUE(foundMyVar);
  EXPECT_TRUE(foundSomeFunction);
  EXPECT_TRUE(foundPrivate);
}

TEST_F(LexerTest, StringParsingAndPeek) {
  auto res = lexer.setSource("data/strings.pic");
  if (!res)
    FAIL() << res.error().message();

  auto peekRes = lexer.peekToken();
  if (!peekRes)
    FAIL() << peekRes.error().message();
  // peek should return first token (identifier 's') without consuming
  EXPECT_EQ(peekRes->type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(peekRes->value(), "s");

  auto nextRes = lexer.nextToken();
  if (!nextRes)
    FAIL() << nextRes.error().message();
  EXPECT_EQ(nextRes->type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(nextRes->value(), "s");

  // advance to string token
  // consume '=', identifier already consumed
  auto eqToken = lexer.nextToken(); // consume '='
  if (!eqToken)
    FAIL() << eqToken.error().message();
  auto strRes = lexer.nextToken();
  if (!strRes)
    FAIL() << strRes.error().message();
  EXPECT_EQ(strRes->type(), picceler::Token::Type::STRING);
  EXPECT_EQ(strRes->value(), "hello world");
}

TEST_F(LexerTest, UnknownCharacterProducesUnknownToken) {
  auto res = lexer.setSource("data/weird.pic");
  if (!res)
    FAIL() << res.error().message();

  auto tokensRes = lexer.tokenizeAll();
  if (!tokensRes)
    FAIL() << tokensRes.error().message();
  auto tokens = tokensRes.value();

  bool foundUnknown = false;
  for (const auto &t : tokens) {
    if (t.type() == picceler::Token::Type::UNKNOWN)
      foundUnknown = true;
  }
  EXPECT_TRUE(foundUnknown);
}
