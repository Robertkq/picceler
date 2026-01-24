#include "lexer.h"
#include <gtest/gtest.h>

class LexerTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  picceler::Lexer lexer;
};

TEST_F(LexerTest, EmptyInput) {
  auto res = lexer.setSource("tests/data/empty.pic");
  if (!res)
    FAIL() << res.error().message();

  auto tokensRes = lexer.tokenizeAll();
  if (!tokensRes)
    FAIL() << tokensRes.error().message();
  auto tokens = tokensRes.value();

  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0]._type, picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, LoadImageStatement) {
  auto res = lexer.setSource("tests/data/load_image.pic");
  if (!res)
    FAIL() << res.error().message();

  // file contents:
  // img = load_image("cat.jpg")
  auto tokensRes = lexer.tokenizeAll();
  if (!tokensRes)
    FAIL() << tokensRes.error().message();
  auto tokens = tokensRes.value();

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

TEST_F(LexerTest, NumbersParsing) {
  auto res = lexer.setSource("tests/data/numbers.pic");
  if (!res)
    FAIL() << res.error().message();

  auto tokensRes = lexer.tokenizeAll();
  if (!tokensRes)
    FAIL() << tokensRes.error().message();
  auto tokens = tokensRes.value();

  std::vector<std::string> numbers;
  for (const auto &t : tokens) {
    if (t._type == picceler::Token::Type::NUMBER)
      numbers.push_back(t._value);
  }

  ASSERT_EQ(numbers.size(), 3);
  EXPECT_EQ(numbers[0], "123");
  EXPECT_EQ(numbers[1], "-3.14");
  EXPECT_EQ(numbers[2], "0.5");
}

TEST_F(LexerTest, KernelTokenSequence) {
  auto res = lexer.setSource("tests/data/kernel.pic");
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
    if (t._type == picceler::Token::Type::SYMBOL)
      seq.push_back(t._value);
    else if (t._type == picceler::Token::Type::NUMBER)
      seq.push_back(t._value);
    else if (t._type == picceler::Token::Type::IDENTIFIER)
      seq.push_back(t._value);
  }

  std::vector<std::string> expected = {"k", "=", "[", "[", "1", ",", "2", "]", ",", "[", "3", ",", "4", "]", "]"};
  // the tokens vector also includes EOF at the end; compare prefix
  ASSERT_GE(seq.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(seq[i], expected[i]);
  }
}

TEST_F(LexerTest, IdentifiersAndUnderscores) {
  auto res = lexer.setSource("tests/data/identifiers.pic");
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
    if (t._type == picceler::Token::Type::IDENTIFIER) {
      if (t._value == "my_var")
        foundMyVar = true;
      if (t._value == "some_function")
        foundSomeFunction = true;
      if (t._value == "_private123")
        foundPrivate = true;
    }
  }
  EXPECT_TRUE(foundMyVar);
  EXPECT_TRUE(foundSomeFunction);
  EXPECT_TRUE(foundPrivate);
}

TEST_F(LexerTest, StringParsingAndPeek) {
  auto res = lexer.setSource("tests/data/strings.pic");
  if (!res)
    FAIL() << res.error().message();

  auto peekRes = lexer.peekToken();
  if (!peekRes)
    FAIL() << peekRes.error().message();
  // peek should return first token (identifier 's') without consuming
  EXPECT_EQ(peekRes->_type, picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(peekRes->_value, "s");

  auto nextRes = lexer.nextToken();
  if (!nextRes)
    FAIL() << nextRes.error().message();
  EXPECT_EQ(nextRes->_type, picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(nextRes->_value, "s");

  // advance to string token
  // consume '=', identifier already consumed
  lexer.nextToken(); // consume '='
  auto strRes = lexer.nextToken();
  if (!strRes)
    FAIL() << strRes.error().message();
  EXPECT_EQ(strRes->_type, picceler::Token::Type::STRING);
  EXPECT_EQ(strRes->_value, "hello world");
}

TEST_F(LexerTest, UnknownCharacterProducesUnknownToken) {
  auto res = lexer.setSource("tests/data/weird.pic");
  if (!res)
    FAIL() << res.error().message();

  auto tokensRes = lexer.tokenizeAll();
  if (!tokensRes)
    FAIL() << tokensRes.error().message();
  auto tokens = tokensRes.value();

  bool foundUnknown = false;
  for (const auto &t : tokens) {
    if (t._type == picceler::Token::Type::UNKNOWN)
      foundUnknown = true;
  }
  EXPECT_TRUE(foundUnknown);
}
