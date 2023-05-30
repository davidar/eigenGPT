#include <fstream>
#include <iostream>

#include "token.hpp"
#include "transformer.hpp"

int main() {
  std::ifstream merge("../gpt2-tokenizer/tokenizer/assets/merges.txt");
  std::ifstream vocab_txt("../gpt2-tokenizer/tokenizer/assets/vocab.txt");
  assert(merge.is_open() && vocab_txt.is_open());
  Tokeniser tokeniser(merge, vocab_txt);
  safetensors::safetensors_t param;
  Transformer transformer(param);

  std::vector<int> prompt =
      tokeniser("Alan Turing theorized that computers would one day become");
  int n_tokens_to_generate = 40;
  std::vector<int> tokens = prompt;
  int total = tokens.size() + n_tokens_to_generate;
  assert(total < n_ctx);
  for (int posn = 0; posn < total; posn++) {
    int token = tokens[posn];
    std::cerr << vocab[token] << std::flush;
    token = transformer(token, posn);
    if (posn + 1 >= tokens.size()) {
      tokens.push_back(token);
    }
  }
}
