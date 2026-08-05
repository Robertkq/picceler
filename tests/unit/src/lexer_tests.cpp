#include "lexer.h"
#include <gtest/gtest.h>

class LexerTest : public ::testing::Test {
protected:
  picceler::Lexer _lexer;

  // Helper to extract tokens or fail the test gracefully if lexing errors out
  std::vector<picceler::Token> tokenize(std::string_view source) {
    _lexer.setSourceString(source);
    auto res = _lexer.getTokens();
    if (!res) {
      ADD_FAILURE() << "Lexing failed unexpectedly: " << res.error().message();
      return {};
    }
    return std::move(*res);
  }
};

// -----------------------------------------------------------------------------
// Basic Tokenizing & Literals
// -----------------------------------------------------------------------------

TEST_F(LexerTest, EmptyInput) {
  auto tokens = tokenize("");

  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, LoadImageStatement) {
  auto tokens = tokenize(R"(img = load_image("cat.jpg"))");

  ASSERT_EQ(tokens.size(), 7);
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

  EXPECT_EQ(tokens[6].type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, NumbersParsing) {
  std::string_view code = R"(
      a = 123
      b = -3.14
      c = 0.5
  )";

  auto tokens = tokenize(code);

  // Exact check: 3 statements (3 tokens each) + 1 EOF = 10 tokens
  ASSERT_EQ(tokens.size(), 10);

  EXPECT_EQ(tokens[0].value(), "a");
  EXPECT_EQ(tokens[2].type(), picceler::Token::Type::NUMBER);
  EXPECT_EQ(tokens[2].value(), "123");

  EXPECT_EQ(tokens[3].value(), "b");
  EXPECT_EQ(tokens[5].type(), picceler::Token::Type::NUMBER);
  EXPECT_EQ(tokens[5].value(), "-3.14");

  EXPECT_EQ(tokens[6].value(), "c");
  EXPECT_EQ(tokens[8].type(), picceler::Token::Type::NUMBER);
  EXPECT_EQ(tokens[8].value(), "0.5");

  EXPECT_EQ(tokens[9].type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, KernelTokenSequence) {
  auto tokens = tokenize(R"(k = [[1, 2], [3, 4]])");

  std::vector<std::string> expectedValues = {"k", "=", "[", "[", "1", ",", "2", "]", ",", "[", "3", ",", "4", "]", "]"};

  ASSERT_EQ(tokens.size(), expectedValues.size() + 1); // +1 for EOF

  for (size_t i = 0; i < expectedValues.size(); ++i) {
    EXPECT_EQ(tokens[i].value(), expectedValues[i]) << "Mismatch at index " << i;
  }
  EXPECT_EQ(tokens.back().type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, IdentifiersAndUnderscores) {
  std::string_view code = R"(
      my_var = some_function()
      other_var = _private123
  )";

  auto tokens = tokenize(code);

  ASSERT_EQ(tokens.size(), 9);

  EXPECT_EQ(tokens[0].type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(tokens[0].value(), "my_var");

  EXPECT_EQ(tokens[2].type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(tokens[2].value(), "some_function");

  EXPECT_EQ(tokens[5].type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(tokens[5].value(), "other_var");

  EXPECT_EQ(tokens[7].type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(tokens[7].value(), "_private123");

  EXPECT_EQ(tokens[8].type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, StringParsingAndPeek) {
  _lexer.setSourceString("s = \"hello world\"");

  auto peekRes = _lexer.peekToken();
  ASSERT_TRUE(peekRes.has_value());
  EXPECT_EQ(peekRes->type(), picceler::Token::Type::IDENTIFIER);
  EXPECT_EQ(peekRes->value(), "s");

  auto t1 = _lexer.nextToken();
  ASSERT_TRUE(t1.has_value());
  EXPECT_EQ(t1->value(), "s");

  auto t2 = _lexer.nextToken();
  ASSERT_TRUE(t2.has_value());
  EXPECT_EQ(t2->type(), picceler::Token::Type::ASSIGN);

  auto t3 = _lexer.nextToken();
  ASSERT_TRUE(t3.has_value());
  EXPECT_EQ(t3->type(), picceler::Token::Type::STRING);
  EXPECT_EQ(t3->value(), "hello world");
}

TEST_F(LexerTest, UnknownCharacterProducesUnknownToken) {
  auto tokens = tokenize("a = @");

  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0].value(), "a");
  EXPECT_EQ(tokens[1].value(), "=");
  EXPECT_EQ(tokens[2].type(), picceler::Token::Type::UNKNOWN);
  EXPECT_EQ(tokens[2].value(), "@");
  EXPECT_EQ(tokens[3].type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, UnaryMinusVsBinaryMinus) {
  auto tokens = tokenize("x = -5 - y");

  // Expected: x, =, -5, -, y, EOF
  ASSERT_EQ(tokens.size(), 6);

  EXPECT_EQ(tokens[0].value(), "x");
  EXPECT_EQ(tokens[1].type(), picceler::Token::Type::ASSIGN);

  // Unary minus with digit gets read as part of NUMBER token by readNumber()
  EXPECT_EQ(tokens[2].type(), picceler::Token::Type::NUMBER);
  EXPECT_EQ(tokens[2].value(), "-5");

  // Binary minus gets read as MINUS token
  EXPECT_EQ(tokens[3].type(), picceler::Token::Type::MINUS);
  EXPECT_EQ(tokens[3].value(), "-");

  EXPECT_EQ(tokens[4].value(), "y");
  EXPECT_EQ(tokens[5].type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, OperatorsWithoutWhitespace) {
  auto tokens = tokenize("a+b*c/d-e");

  std::vector<picceler::Token::Type> expectedTypes = {picceler::Token::Type::IDENTIFIER, // a
                                                      picceler::Token::Type::PLUS,       // +
                                                      picceler::Token::Type::IDENTIFIER, // b
                                                      picceler::Token::Type::MULTIPLY,   // *
                                                      picceler::Token::Type::IDENTIFIER, // c
                                                      picceler::Token::Type::DIVIDE,     // /
                                                      picceler::Token::Type::IDENTIFIER, // d
                                                      picceler::Token::Type::MINUS,      // -
                                                      picceler::Token::Type::IDENTIFIER, // e
                                                      picceler::Token::Type::EOF_TOKEN};

  ASSERT_EQ(tokens.size(), expectedTypes.size());
  for (size_t i = 0; i < expectedTypes.size(); ++i) {
    EXPECT_EQ(tokens[i].type(), expectedTypes[i]) << "Mismatch at index " << i;
  }
}

TEST_F(LexerTest, ArrowVsMinusToken) {
  auto tokens = tokenize("def foo() -> int64 { return a - b }");

  // Check that '->' is ARROW and '-' is MINUS
  bool foundArrow = false;
  bool foundMinus = false;

  for (const auto &tok : tokens) {
    if (tok.type() == picceler::Token::Type::ARROW)
      foundArrow = true;
    if (tok.type() == picceler::Token::Type::MINUS)
      foundMinus = true;
  }

  EXPECT_TRUE(foundArrow);
  EXPECT_TRUE(foundMinus);
}

// -----------------------------------------------------------------------------
// Comment Handling
// -----------------------------------------------------------------------------

TEST_F(LexerTest, FullLineCommentsAreIgnored) {
  std::string_view code = R"(
# This is a full line comment
a = 1
      # Another comment
b = 2
)";

  auto tokens = tokenize(code);

  std::vector<std::string> expectedValues = {"a", "=", "1", "b", "=", "2"};
  ASSERT_EQ(tokens.size(), expectedValues.size() + 1);

  for (size_t i = 0; i < expectedValues.size(); ++i) {
    EXPECT_EQ(tokens[i].value(), expectedValues[i]);
  }
  EXPECT_EQ(tokens.back().type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, ConsecutiveComments) {
  std::string_view code = R"(
# Comment 1
# Comment 2
# Comment 3
a = 5
)";

  auto tokens = tokenize(code);

  ASSERT_EQ(tokens.size(), 4); // a, =, 5, EOF
  EXPECT_EQ(tokens[0].value(), "a");
  EXPECT_EQ(tokens[1].value(), "=");
  EXPECT_EQ(tokens[2].value(), "5");
  EXPECT_EQ(tokens[3].type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, CommentAtEndOfFile) {
  std::string_view code = R"(
a = 1 
# This is the last line, no newline at the end
)";

  auto tokens = tokenize(code);

  ASSERT_EQ(tokens.size(), 4); // a, =, 1, EOF
  EXPECT_EQ(tokens[0].value(), "a");
  EXPECT_EQ(tokens[1].value(), "=");
  EXPECT_EQ(tokens[2].value(), "1");
  EXPECT_EQ(tokens[3].type(), picceler::Token::Type::EOF_TOKEN);
}

TEST_F(LexerTest, InlineComments) {
  std::string_view code = R"(
    img = load_image("cat.jpg") # load the cat image
    blurred = blur(img, 3)      # apply blur
  )";

  auto tokens = tokenize(code);

  std::vector<std::string> expectedValues = {"img", "=",    "load_image", "(",   "cat.jpg", ")", "blurred",
                                             "=",   "blur", "(",          "img", ",",       "3", ")"};

  ASSERT_EQ(tokens.size(), expectedValues.size() + 1);

  for (size_t i = 0; i < expectedValues.size(); ++i) {
    EXPECT_EQ(tokens[i].value(), expectedValues[i]);
  }
  EXPECT_EQ(tokens.back().type(), picceler::Token::Type::EOF_TOKEN);
}

// -----------------------------------------------------------------------------
// Source Location Tracking
// -----------------------------------------------------------------------------

TEST_F(LexerTest, SourceLocationTracking) {
  std::string_view code = "a = 1\nb = 2";

  auto tokens = tokenize(code);

  ASSERT_EQ(tokens.size(), 7); // a, =, 1, b, =, 2, EOF

  // Line 1
  EXPECT_EQ(tokens[0].location().line(), 1);
  EXPECT_EQ(tokens[0].location().column(), 1);

  EXPECT_EQ(tokens[1].location().line(), 1);
  EXPECT_EQ(tokens[1].location().column(), 3);

  // Line 2
  EXPECT_EQ(tokens[3].location().line(), 2);
  EXPECT_EQ(tokens[3].location().column(), 1);
}