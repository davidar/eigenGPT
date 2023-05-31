#include <fstream>
#include <iostream>

#include "transformer.hpp"

int main() {
  int seed = time(NULL);
  std::cerr << "seed: " << seed << std::endl;
  srand(seed);

  Transformer transformer;
  int token = n_vocab - 1;
  for (int posn = 0; posn < n_ctx; posn++) {
    token = transformer(token, posn);
    std::cerr << vocab[token] << std::flush;
  }
}
