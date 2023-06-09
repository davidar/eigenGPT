#include <filesystem>
#include <fstream>
#include <iostream>

#include "token.hpp"
#include "transformer.hpp"

namespace fs = std::filesystem;

int main() {
  std::ifstream merge("../gpt2-tokenizer/tokenizer/assets/merges.txt");
  std::ifstream vocab("../gpt2-tokenizer/tokenizer/assets/vocab.txt");
  std::ifstream model("../gpt2/model.safetensors", std::ios::binary);
  assert(merge.is_open() && vocab.is_open() && model.is_open());
  Tokeniser tokeniser(merge, vocab);
  auto param = safetensors::safetensors_t(model);

  std::vector<TransformerBlock> blocks;
  for (int b = 0; b < n_layer; b++) {
    std::cerr << 100 * b / n_layer << "%\r" << std::flush;
    blocks.emplace_back(param, b);
  }

  auto wte = param.matrix("wte.weight");
  auto wpe = param.matrix("wpe.weight");
  auto w_ln = param.vector("ln_f.weight") * sqrt(n_embd);
  auto b_ln = param.vector("ln_f.bias");

  std::vector<int> prompt =
      tokeniser("Alan Turing theorized that computers would one day become");
  int n_tokens_to_generate = 40;
  std::vector<int> tokens = prompt;
  int total = tokens.size() + n_tokens_to_generate;
  assert(total < n_ctx);
  for (int posn = 0; posn < total; posn++) {
    int token = tokens[posn];
    std::cerr << tokeniser(std::vector<int>{token}) << std::flush;
    Embedding x = wte.row(token) + wpe.row(posn);
    for (auto &block : blocks) {
      block(x);
    }
    x = TransformerBlock::norm(x);
    x.array() *= w_ln.array();
    x += b_ln;
    auto logits = wte * x;
    logits.maxCoeff(&token);
    if (posn + 1 >= tokens.size()) {
      tokens.push_back(token);
    }
  }
}
