#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <Eigen/Dense>
#include <fmt/core.h>
#include <unsupported/Eigen/SpecialFunctions>

#include "safetensors.hpp"

using namespace Eigen;
using fmt::format;

constexpr size_t n_ctx = 1024;
constexpr size_t n_embd = 768;
constexpr size_t n_head = 12;
constexpr size_t n_layer = 12;
constexpr size_t n_vocab = 50257;
constexpr size_t D = n_embd / n_head;

using Embedding = Vector<float, n_embd>;

class TransformerBlock {
public:
  float kv[n_ctx][2 * n_embd];
  int n_seq = 0;

  float w_attn1[n_embd * 3 * n_embd];
  float b_attn1[3 * n_embd];

  float w_attn2[n_embd * n_embd];
  float b_attn2[n_embd];

  float w_mlp1[n_embd * 4 * n_embd];
  float b_mlp1[4 * n_embd];

  float w_mlp2[4 * n_embd * n_embd];
  float b_mlp2[n_embd];

  float w_ln1[n_embd];
  float b_ln1[n_embd];
  float w_ln2[n_embd];
  float b_ln2[n_embd];

  TransformerBlock(safetensors::safetensors_t param, int b) {
    memcpy(w_attn1, param.data(format("h.{}.attn.c_attn.weight", b)), sizeof(w_attn1));
    memcpy(b_attn1, param.data(format("h.{}.attn.c_attn.bias", b)), sizeof(b_attn1));
    memcpy(w_attn2, param.data(format("h.{}.attn.c_proj.weight", b)), sizeof(w_attn2));
    memcpy(b_attn2, param.data(format("h.{}.attn.c_proj.bias", b)), sizeof(b_attn2));
    memcpy(w_mlp1, param.data(format("h.{}.mlp.c_fc.weight", b)), sizeof(w_mlp1));
    memcpy(b_mlp1, param.data(format("h.{}.mlp.c_fc.bias", b)), sizeof(b_mlp1));
    memcpy(w_mlp2, param.data(format("h.{}.mlp.c_proj.weight", b)), sizeof(w_mlp2));
    memcpy(b_mlp2, param.data(format("h.{}.mlp.c_proj.bias", b)), sizeof(b_mlp2));
    memcpy(b_ln1, param.data(format("h.{}.ln_1.bias", b)), sizeof(b_ln1));
    memcpy(w_ln1, param.data(format("h.{}.ln_1.weight", b)), sizeof(w_ln1));
    memcpy(b_ln2, param.data(format("h.{}.ln_2.bias", b)), sizeof(b_ln2));
    memcpy(w_ln2, param.data(format("h.{}.ln_2.weight", b)), sizeof(w_ln2));
  }

  static void norm(float x[n_embd]) {
    float sum = 0;
    for (int i = 0; i < n_embd; i++)
      sum += x[i];
    float mean = sum / n_embd;
    for (int i = 0; i < n_embd; i++)
      x[i] -= mean;
    float norm = 0;
    for (int i = 0; i < n_embd; i++)
      norm += x[i] * x[i];
    norm = sqrt(norm);
    for (int i = 0; i < n_embd; i++)
      x[i] /= norm;
  }

  void operator()(float x[n_embd]) {
    float norm_x[n_embd];
    memcpy(norm_x, x, sizeof(float) * n_embd);
    norm(norm_x);

    // update kv cache
    float qkv_x[3 * n_embd] = {0};
    for (int i = 0; i < 3 * n_embd; i++) {
      for (int j = 0; j < n_embd; j++) {
        qkv_x[i] += w_attn1[j * (3 * n_embd) + i] * (b_ln1[j] + w_ln1[j] * sqrt(n_embd) * norm_x[j]);
      }
      qkv_x[i] += b_attn1[i];
      if (i >= n_embd)
        kv[n_seq][i - n_embd] = qkv_x[i];
    }
    n_seq += 1;

    // attention
    float attn[n_embd] = {0};
    float asum[n_head] = {0};
    for (int head = 0; head < n_head; head++) {
      for (int posn = 0; posn < n_seq; posn++) {
        float a = 0;
        for (int d = 0; d < D; d++) {
          a += kv[posn][d + D * head] * qkv_x[d + D * head];
        }
        a = std::exp(a / sqrt(D));
        for (int d = 0; d < D; d++) {
          attn[d + D * head] += kv[posn][d + D * head + n_embd] * a;
        }
        asum[head] += a;
      }
    }

    for (int i = 0; i < n_embd; i++) {
      for (int j = 0; j < n_embd; j++) {
        x[i] += w_attn2[j * n_embd + i] * attn[j] / asum[j / D];
      }
      x[i] += b_attn2[i];
    }

    // mlp
    float h[4 * n_embd] = {0};
    memcpy(norm_x, x, sizeof(float) * n_embd);
    norm(norm_x);

    for (int i = 0; i < 4 * n_embd; i++) {
      for (int j = 0; j < n_embd; j++) {
        h[i] += w_mlp1[j * (4 * n_embd) + i] * (b_ln2[j] + w_ln2[j] * sqrt(n_embd) * norm_x[j]);
      }
      h[i] += b_mlp1[i];
      h[i] *= (1 + std::erf(h[i] / sqrt(2))) / 2;
    }

    for (int i = 0; i < n_embd; i++) {
      for (int j = 0; j < 4 * n_embd; j++) {
        x[i] += w_mlp2[j * n_embd + i] * h[j];
      }
      x[i] += b_mlp2[i];
    }
  }
};
