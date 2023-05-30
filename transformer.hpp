#include <fmt/core.h>

#include "safetensors.hpp"

using fmt::format;

constexpr size_t n_ctx = 1024;
constexpr size_t n_embd = 768;
constexpr size_t n_head = 12;
constexpr size_t n_layer = 12;
constexpr size_t n_vocab = 50257;
constexpr size_t D = n_embd / n_head;

#define FOR_EMBED(var, mul) for (int var = 0; var < mul * n_embd; var++)

class TransformerBlock {
public:
  float kv[n_ctx][2 * n_embd];
  int n_seq = 0;
  float *w_attn1, *b_attn1, *w_attn2, *b_attn2, *w_mlp1, *b_mlp1, *w_mlp2,
      *b_mlp2, *w_ln1, *b_ln1, *w_ln2, *b_ln2;

  TransformerBlock(safetensors::safetensors_t param, int b)
      : w_attn1(param.data(format("h.{}.attn.c_attn.weight", b))),
        b_attn1(param.data(format("h.{}.attn.c_attn.bias", b))),
        w_attn2(param.data(format("h.{}.attn.c_proj.weight", b))),
        b_attn2(param.data(format("h.{}.attn.c_proj.bias", b))),
        w_mlp1(param.data(format("h.{}.mlp.c_fc.weight", b))),
        b_mlp1(param.data(format("h.{}.mlp.c_fc.bias", b))),
        w_mlp2(param.data(format("h.{}.mlp.c_proj.weight", b))),
        b_mlp2(param.data(format("h.{}.mlp.c_proj.bias", b))),
        b_ln1(param.data(format("h.{}.ln_1.bias", b))),
        w_ln1(param.data(format("h.{}.ln_1.weight", b))),
        b_ln2(param.data(format("h.{}.ln_2.bias", b))),
        w_ln2(param.data(format("h.{}.ln_2.weight", b))) {}

  void operator()(float x[n_embd]) {
    float norm_x[n_embd];
    float sum = 0, sqnorm = 0;
    FOR_EMBED(i, 1) sum += x[i];
    FOR_EMBED(i, 1) {
      norm_x[i] = x[i] - sum / n_embd;
      sqnorm += norm_x[i] * norm_x[i];
    }

    // update kv cache
    float q[n_embd];
    FOR_EMBED(i, 3) {
      float *qkv = i < n_embd ? &q[i] : &kv[n_seq][i - n_embd];
      *qkv = b_attn1[i];
      FOR_EMBED(j, 1) {
        *qkv += w_attn1[j * (3 * n_embd) + i] *
                (b_ln1[j] + w_ln1[j] * sqrt(n_embd) * norm_x[j] / sqrt(sqnorm));
      }
    }
    n_seq++;

    // attention
    float attn[n_embd] = {0};
    float asum[n_head] = {0};
    for (int head = 0; head < n_head; head++) {
      for (int posn = 0; posn < n_seq; posn++) {
        float a = 0;
        for (int d = D * head; d < D + D * head; d++) {
          a += kv[posn][d] * q[d];
        }
        asum[head] += a = std::exp(a / sqrt(D));
        for (int d = D * head; d < D + D * head; d++) {
          attn[d] += kv[posn][d + n_embd] * a;
        }
      }
    }

    FOR_EMBED(i, 1) {
      float r = b_attn2[i];
      FOR_EMBED(j, 1) r += w_attn2[j * n_embd + i] * attn[j] / asum[j / D];
      x[i] += r;
      sum += r;
    }

    sqnorm = 0;
    FOR_EMBED(i, 1) {
      norm_x[i] = x[i] - sum / n_embd;
      sqnorm += norm_x[i] * norm_x[i];
    }

    // mlp
    float h[4 * n_embd];
    FOR_EMBED(i, 4) {
      h[i] = b_mlp1[i];
      FOR_EMBED(j, 1) {
        h[i] += w_mlp1[j * (4 * n_embd) + i] *
                (b_ln2[j] + w_ln2[j] * sqrt(n_embd) * norm_x[j] / sqrt(sqnorm));
      }
      h[i] *= (1 + std::erf(h[i] / sqrt(2))) / 2;
    }

    FOR_EMBED(i, 1) {
      FOR_EMBED(j, 4) x[i] += w_mlp2[j * n_embd + i] * h[j];
      x[i] += b_mlp2[i];
    }
  }
};
