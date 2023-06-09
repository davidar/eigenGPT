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
  // (k,v) cache, rows are added for each token
  Matrix<float, Dynamic, 2 * n_embd> kv;

  Matrix<float, 3 * n_embd, n_embd> w_attn1;
  Vector<float, 3 * n_embd> b_attn1;

  Matrix<float, n_embd, n_embd> w_attn2;
  Vector<float, n_embd> b_attn2;

  Matrix<float, 4 * n_embd, n_embd> w_mlp1;
  Vector<float, 4 * n_embd> b_mlp1;

  Matrix<float, n_embd, 4 * n_embd> w_mlp2;
  Vector<float, n_embd> b_mlp2;

  TransformerBlock(safetensors::safetensors_t param, int b)
      : kv(0, 2 * n_embd),
        w_attn1(param.matrix(format("h.{}.attn.c_attn.weight", b)).transpose()),
        b_attn1(param.vector(format("h.{}.attn.c_attn.bias", b))),
        w_attn2(param.matrix(format("h.{}.attn.c_proj.weight", b)).transpose()),
        b_attn2(param.vector(format("h.{}.attn.c_proj.bias", b))),
        w_mlp1(param.matrix(format("h.{}.mlp.c_fc.weight", b)).transpose()),
        b_mlp1(param.vector(format("h.{}.mlp.c_fc.bias", b))),
        w_mlp2(param.matrix(format("h.{}.mlp.c_proj.weight", b)).transpose()),
        b_mlp2(param.vector(format("h.{}.mlp.c_proj.bias", b))) {

    // bake the normalisation constants into the weights
    b_attn1 += w_attn1 * param.vector(format("h.{}.ln_1.bias", b));
    w_attn1.array().rowwise() *=
        param.vector(format("h.{}.ln_1.weight", b)).array().transpose() *
        sqrt(n_embd);
    b_mlp1 += w_mlp1 * param.vector(format("h.{}.ln_2.bias", b));
    w_mlp1.array().rowwise() *=
        param.vector(format("h.{}.ln_2.weight", b)).array().transpose() *
        sqrt(n_embd);
  }

  void operator()(Embedding &x) {
    // update kv cache
    auto n_seq = kv.rows();
    Vector<float, 3 *n_embd> qkv_x = w_attn1 * norm(x) + b_attn1;
    kv.conservativeResize(n_seq + 1, NoChange);
    kv.row(n_seq) = qkv_x.segment<2 * n_embd>(n_embd);

    // attention
    Embedding attn;
    for (int i = 0; i < n_head; i++) {
      Vector<float, D> q = qkv_x.segment<D>(D * i);
      auto k = kv.middleCols<D>(D * i);
      auto v = kv.middleCols<D>(D * i + n_embd);
      VectorXf a = (k * q / sqrt(D)).array().exp();
      a /= a.sum();
      attn.segment<D>(D * i) = v.transpose() * a;
    }
    x += w_attn2 * attn + b_attn2;

    // mlp
    Vector<float, 4 *n_embd> h = w_mlp1 * norm(x) + b_mlp1;
    h.array() *= (1 + (h / sqrt(2)).array().erf()) / 2; // gelu
    x += w_mlp2 * h + b_mlp2;
  }

  static Embedding norm(Embedding x) {
    x.array() -= x.mean();
    x.normalize();
    return x;
  }
};
