#include "lexer.h"
#include <gtest/gtest.h>

class LexerTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  picceler::Lexer _lexer;
};

TEST_F(LexerTest, EmptyInput) {
  _lexer.setSourceString(R"()");

  auto tokensRes = _lexer.tokenizeAll();
  ASSERT_TRUE(tokensRes.has_value());
  const auto &tokens = tokensRes.value();

  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, LoadImageStatement) {
  _lexer.setSourceString(R"(img = load_image("cat.jpg"))");

  auto tokensRes = _lexer.tokenizeAll();
  ASSERT_TRUE(tokensRes.has_value());

  const auto &tokens = tokensRes.value();
  ASSERT_GE(tokens.size(), 7);
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
  _lexer.setSourceString(R"(
      a = 123
      b = -3.14
      c = 0.5
  )");

  auto tokensRes = _lexer.tokenizeAll();

  ASSERT_TRUE(tokensRes.has_value());
  const auto &tokens = tokensRes.value();

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
  _lexer.setSourceString(R"(k = [[1, 2], [3, 4]])");

  auto tokensRes = _lexer.tokenizeAll();
  ASSERT_TRUE(tokensRes.has_value());
  const auto &tokens = tokensRes.value();

  // Expect sequence: IDENTIFIER, '=', '[', '[', NUMBER(1), ',', NUMBER(2), ']', ',', '[', NUMBER(3), ',', NUMBER(4),
  // ']', ']', EOF
  std::vector<std::string> seq;
  for (const auto &t : tokens) {
    switch (t.type()) {
    case picceler::Token::Type::IDENTIFIER:
    case picceler::Token::Type::ASSIGN:
    case picceler::Token::Type::L_BRACKET:
    case picceler::Token::Type::R_BRACKET:
    case picceler::Token::Type::COMMA:
    case picceler::Token::Type::NUMBER:
      seq.push_back(t.value());
      break;
    default:
      break;
    }
  }

  std::vector<std::string> expected = {"k", "=", "[", "[", "1", ",", "2", "]", ",", "[", "3", ",", "4", "]", "]"};
  // the tokens vector also includes EOF at the end; compare prefix
  ASSERT_GE(seq.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(seq[i], expected[i]);
  }
}

TEST_F(LexerTest, IdentifiersAndUnderscores) {
  _lexer.setSourceString(R"(
        my_var = some_function()
        other_var = _private123
    )");

  auto tokensRes = _lexer.tokenizeAll();
  ASSERT_TRUE(tokensRes.has_value());
  const auto &tokens = tokensRes.value();

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
  _lexer.setSourceString(R"(
        s = "hello world"
        empty = ""
    )");

  auto peekRes = _lexer.peekToken();
  ASSERT_TRUE(peekRes.has_value());
  // peek should return first token (identifier 's') without consuming
  EXPECT_EQ(peekRes->type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(peekRes->value(), "s");

  auto nextRes = _lexer.nextToken();
  if (!nextRes)
    FAIL() << nextRes.error().message();
  EXPECT_EQ(nextRes->type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(nextRes->value(), "s");

  // advance to string token
  // consume '=', identifier already consumed
  auto eqToken = _lexer.nextToken(); // consume '='
  if (!eqToken)
    FAIL() << eqToken.error().message();
  auto strRes = _lexer.nextToken();
  if (!strRes)
    FAIL() << strRes.error().message();
  EXPECT_EQ(strRes->type(), picceler::Token::Type::STRING);
  EXPECT_EQ(strRes->value(), "hello world");
}

TEST_F(LexerTest, UnknownCharacterProducesUnknownToken) {
  _lexer.setSourceString("a = @");

  auto tokensRes = _lexer.tokenizeAll();
  ASSERT_TRUE(tokensRes.has_value());
  const auto &tokens = tokensRes.value();

  bool foundUnknown = false;
  for (const auto &t : tokens) {
    if (t.type() == picceler::Token::Type::UNKNOWN)
      foundUnknown = true;
  }
  EXPECT_TRUE(foundUnknown);
}

TEST_F(LexerTest, FullLineCommentsAreIgnored) {
  _lexer.setSourceString(R"(
# This is a full line comment
a = 1
      # Another comment
b = 2
)");

  auto tokensRes = _lexer.tokenizeAll();
  ASSERT_TRUE(tokensRes.has_value());
  const auto &tokens = tokensRes.value();

  std::vector<std::string> expected = {"a", "=", "1", "b", "=", "2"};
  std::vector<std::string> actual;

  for (const auto &t : tokens) {
    if (t.type() != picceler::Token::Type::EOF_TOKEN) {
      actual.push_back(t.value());
    }
  }

  EXPECT_EQ(actual, expected);
}

TEST_F(LexerTest, ConsecutiveComments) {
  _lexer.setSourceString(R"(
# Comment 1
# Comment 2
# Comment 3
a = 5
)");

  auto tokensRes = _lexer.tokenizeAll();
  ASSERT_TRUE(tokensRes.has_value());
  const auto &tokens = tokensRes.value();

  std::vector<std::string> actual;
  for (const auto &t : tokens) {
    if (t.type() != picceler::Token::Type::EOF_TOKEN) {
      actual.push_back(t.value());
    }
  }

  std::vector<std::string> expected = {"a", "=", "5"};
  EXPECT_EQ(actual, expected);
}

TEST_F(LexerTest, CommentAtEndOfFile) {
  _lexer.setSourceString(R"(
a = 1 
# This is the last line, no newline at the end
)");

  auto tokensRes = _lexer.tokenizeAll();
  ASSERT_TRUE(tokensRes.has_value());
  const auto &tokens = tokensRes.value();

  std::vector<std::string> actual;
  for (const auto &t : tokens) {
    if (t.type() != picceler::Token::Type::EOF_TOKEN) {
      actual.push_back(t.value());
    }
  }

  std::vector<std::string> expected = {"a", "=", "1"};
  EXPECT_EQ(actual, expected);
}

TEST_F(LexerTest, InlineComments) {
  _lexer.setSourceString(R"(
    img = load_image("cat.jpg") # load the cat image
    blurred = blur(img, 3)      # apply blur
  )");

  auto tokensRes = _lexer.tokenizeAll();
  ASSERT_TRUE(tokensRes.has_value());
  const auto &tokens = tokensRes.value();

  std::vector<std::string> values;
  for (const auto &t : tokens) {
    if (t.type() != picceler::Token::Type::EOF_TOKEN) {
      values.push_back(t.value());
    }
  }

  std::vector<std::string> expected = {"img", "=",    "load_image", "(",   "cat.jpg", ")", "blurred",
                                       "=",   "blur", "(",          "img", ",",       "3", ")"};

  EXPECT_EQ(values, expected);
}
