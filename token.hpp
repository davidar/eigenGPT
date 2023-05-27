#include "gpt2-tokenizer/tokenizer/bpe.h"

class Tokeniser {
  RE2 re;
  BPERanks bpe_ranks;
  std::unordered_map<uint8_t, wchar_t> b2u;
  std::unordered_map<wchar_t, uint8_t> u2b;
  std::unordered_map<std::string, int> t2i;
  std::unordered_map<int, std::string> i2t;

public:
  Tokeniser(std::istream &merges, std::istream &vocab_txt)
      : re("('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
           "?[^\\s\\p{L}\\p{N}]+|\\s+\\(?!\\S\\)|\\s+)") {
    load_merge_rules(merges, &bpe_ranks);
    bytes_to_unicode(&b2u, &u2b);
    load_vocab(vocab_txt, &t2i, &i2t);
  }

  std::vector<int> operator()(std::string s) {
    std::vector<int> ids;
    encode(s, re, bpe_ranks, b2u, t2i, &ids);
    return ids;
  }

  std::string operator()(std::vector<int> ids) { return decode(ids, u2b, i2t); }
};
