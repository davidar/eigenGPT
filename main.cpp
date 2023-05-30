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

  /*
  auto print_offset = [&](std::string name) {
    std::cout << param.meta.at(name).data_offsets.first << ", // " << name
              << std::endl;
  };
  print_offset("wte.weight");
  print_offset("wpe.weight");
  print_offset("ln_f.weight");
  print_offset("ln_f.bias");
  for (int i = 0; i < n_layer; i++) {
    std::cout << "{ // layer " << i << std::endl;
    print_offset("h." + std::to_string(i) + ".attn.c_attn.weight");
    print_offset("h." + std::to_string(i) + ".attn.c_attn.bias");
    print_offset("h." + std::to_string(i) + ".attn.c_proj.weight");
    print_offset("h." + std::to_string(i) + ".attn.c_proj.bias");
    print_offset("h." + std::to_string(i) + ".ln_1.weight");
    print_offset("h." + std::to_string(i) + ".ln_1.bias");
    print_offset("h." + std::to_string(i) + ".mlp.c_fc.weight");
    print_offset("h." + std::to_string(i) + ".mlp.c_fc.bias");
    print_offset("h." + std::to_string(i) + ".mlp.c_proj.weight");
    print_offset("h." + std::to_string(i) + ".mlp.c_proj.bias");
    print_offset("h." + std::to_string(i) + ".ln_2.weight");
    print_offset("h." + std::to_string(i) + ".ln_2.bias");
    std::cout << "}," << std::endl;
  }
  */

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
