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
  int kv_size;

  float w_attn1[3 * n_embd][n_embd];
  float b_attn1[3 * n_embd];

  float w_attn2[n_embd][n_embd];
  float b_attn2[n_embd];

  float w_mlp1[4 * n_embd][n_embd];
  float b_mlp1[4 * n_embd];

  float w_mlp2[n_embd][4 * n_embd];
  float b_mlp2[n_embd];

  TransformerBlock(safetensors::safetensors_t param, int b) {
    kv_size = 0;

    float *raw_data;

    // Copy w_attn1
    raw_data = param.data(format("h.{}.attn.c_attn.weight", b));
    for (int i = 0; i < 3 * n_embd; i++)
      for (int j = 0; j < n_embd; j++)
        w_attn1[i][j] = raw_data[j * (3 * n_embd) + i]; // Note the transpose

    // Copy b_attn1
    raw_data = param.data(format("h.{}.attn.c_attn.bias", b));
    memcpy(b_attn1, raw_data, sizeof(b_attn1));

    // Copy w_attn2
    raw_data = param.data(format("h.{}.attn.c_proj.weight", b));
    for (int i = 0; i < n_embd; i++)
      for (int j = 0; j < n_embd; j++)
        w_attn2[i][j] = raw_data[j * n_embd + i]; // Note the transpose

    // Copy b_attn2
    raw_data = param.data(format("h.{}.attn.c_proj.bias", b));
    memcpy(b_attn2, raw_data, sizeof(b_attn2));

    // Copy w_mlp1
    raw_data = param.data(format("h.{}.mlp.c_fc.weight", b));
    for (int i = 0; i < 4 * n_embd; i++)
      for (int j = 0; j < n_embd; j++)
        w_mlp1[i][j] = raw_data[j * (4 * n_embd) + i]; // Note the transpose

    // Copy b_mlp1
    raw_data = param.data(format("h.{}.mlp.c_fc.bias", b));
    memcpy(b_mlp1, raw_data, sizeof(b_mlp1));

    // Copy w_mlp2
    raw_data = param.data(format("h.{}.mlp.c_proj.weight", b));
    for (int i = 0; i < n_embd; i++)
      for (int j = 0; j < 4 * n_embd; j++)
        w_mlp2[i][j] = raw_data[j * n_embd + i]; // Note the transpose

    // Copy b_mlp2
    raw_data = param.data(format("h.{}.mlp.c_proj.bias", b));
    memcpy(b_mlp2, raw_data, sizeof(b_mlp2));

    // bake the normalisation constants into the weights
    /*
    raw_data = param.data(format("h.{}.ln_1.bias", b));
    for (int i = 0; i < 3 * n_embd; i++) {
      float sum = 0.0f;
      for (int j = 0; j < n_embd; j++)
        sum += w_attn1[i][j] * raw_data[j];
      b_attn1[i] += sum;
    }

    raw_data = param.data(format("h.{}.ln_1.weight", b));
    for (int i = 0; i < 3 * n_embd; i++)
      for (int j = 0; j < n_embd; j++)
        w_attn1[i][j] *= raw_data[j] * sqrt(n_embd);

    raw_data = param.data(format("h.{}.ln_2.bias", b));
    for (int i = 0; i < 4 * n_embd; i++) {
      float sum = 0.0f;
      for (int j = 0; j < n_embd; j++)
        sum += w_mlp1[i][j] * raw_data[j];
      b_mlp1[i] += sum;
    }

    raw_data = param.data(format("h.{}.ln_2.weight", b));
    for (int i = 0; i < 4 * n_embd; i++)
      for (int j = 0; j < n_embd; j++)
        w_mlp1[i][j] *= raw_data[j] * sqrt(n_embd);
    */
    raw_data = param.data(format("h.{}.ln_1.bias", b));
    for (int i = 0; i < 3 * n_embd; i++) {
      for (int j = 0; j < n_embd; j++)
        b_attn1[i] += w_attn1[i][j] * raw_data[j];
    }

    raw_data = param.data(format("h.{}.ln_1.weight", b));
    for (int i = 0; i < 3 * n_embd; i++)
      for (int j = 0; j < n_embd; j++)
        w_attn1[i][j] *= raw_data[j] * sqrt(n_embd);

    raw_data = param.data(format("h.{}.ln_2.bias", b));
    for (int i = 0; i < 4 * n_embd; i++) {
      for (int j = 0; j < n_embd; j++)
        b_mlp1[i] += w_mlp1[i][j] * raw_data[j];
    }

    raw_data = param.data(format("h.{}.ln_2.weight", b));
    for (int i = 0; i < 4 * n_embd; i++)
      for (int j = 0; j < n_embd; j++)
        w_mlp1[i][j] *= raw_data[j] * sqrt(n_embd);
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

    float qkv_x[3 * n_embd] = {0};
    for (int i = 0; i < 3 * n_embd; i++) {
      for (int j = 0; j < n_embd; j++) {
        qkv_x[i] += w_attn1[i][j] * norm_x[j];
      }
      qkv_x[i] += b_attn1[i];
    }

    // update kv cache
    for (int i = 0; i < 2 * n_embd; i++) {
      kv[kv_size][i] = qkv_x[n_embd + i];
    }
    kv_size += 1;

    // attention
    float attn[n_embd] = {0};
    for (int i = 0; i < n_head; i++) {
      float q[D] = {0};
      float a[n_ctx] = {0};
      float a_sum = 0.0;

      for (int j = 0; j < D; j++) {
        q[j] = qkv_x[j + D * i];
      }

      float k[n_ctx][D] = {0};
      float v[n_ctx][D] = {0};

      for (int r = 0; r < kv_size; r++) {
        for (int c = 0; c < D; c++) {
          k[r][c] = kv[r][c + D * i];
          v[r][c] = kv[r][c + D * i + n_embd];
        }
      }

      for (int j = 0; j < kv_size; j++) {
        for (int l = 0; l < D; l++) {
          a[j] += k[j][l] * q[l];
        }
        a[j] = std::exp(a[j] / sqrt(D));
        a_sum += a[j];
      }

      for (int j = 0; j < kv_size; j++) {
        a[j] /= a_sum;
      }

      for (int j = 0; j < D; j++) {
        for (int l = 0; l < kv_size; l++) {
          attn[j + D * i] += v[l][j] * a[l];
        }
      }
    }

    for (int i = 0; i < n_embd; i++) {
      for (int j = 0; j < n_embd; j++) {
        x[i] += w_attn2[i][j] * attn[j];
      }
      x[i] += b_attn2[i];
    }

    // mlp
    float h[4 * n_embd] = {0};
    memcpy(norm_x, x, sizeof(float) * n_embd);
    norm(norm_x);

    for (int i = 0; i < 4 * n_embd; i++) {
      for (int j = 0; j < n_embd; j++) {
        h[i] += w_mlp1[i][j] * norm_x[j];
      }
      h[i] += b_mlp1[i];
      h[i] = h[i] * (1 + std::erf(h[i] / sqrt(2))) / 2;
    }

    for (int i = 0; i < n_embd; i++) {
      for (int j = 0; j < 4 * n_embd; j++) {
        x[i] += w_mlp2[i][j] * h[j];
      }
      x[i] += b_mlp2[i];
    }
  }
};
