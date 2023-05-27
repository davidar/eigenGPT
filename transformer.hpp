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

class TransformerBlock {
public:
  MatrixXf /*<float, -1, 3 * n_embd>*/ qkv;

  MatrixXf /*<float, n_embd, 3 * n_embd>*/ w_attn1;
  VectorXf /*<float, 3 * n_embd>*/ b_attn1;

  MatrixXf /*<float, n_embd, n_embd>*/ w_attn2;
  VectorXf /*<float, n_embd>*/ b_attn2;

  MatrixXf /*<float, n_embd, 4 * n_embd>*/ w_mlp1;
  VectorXf /*<float, 4 * n_embd>*/ b_mlp1;

  MatrixXf /*<float, 4 * n_embd, n_embd>*/ w_mlp2;
  VectorXf /*<float, n_embd>*/ b_mlp2;

  TransformerBlock(safetensors::safetensors_t param, int b)
      : qkv(0, 3 * n_embd),
        w_attn1(param.matrix(format("h.{}.attn.c_attn.weight", b))),
        b_attn1(param.vector(format("h.{}.attn.c_attn.bias", b))),
        w_attn2(param.matrix(format("h.{}.attn.c_proj.weight", b))),
        b_attn2(param.vector(format("h.{}.attn.c_proj.bias", b))),
        w_mlp1(param.matrix(format("h.{}.mlp.c_fc.weight", b))),
        b_mlp1(param.vector(format("h.{}.mlp.c_fc.bias", b))),
        w_mlp2(param.matrix(format("h.{}.mlp.c_proj.weight", b))),
        b_mlp2(param.vector(format("h.{}.mlp.c_proj.bias", b))) {
    b_attn1 += param.vector(format("h.{}.ln_1.bias", b)).transpose() * w_attn1;
    w_attn1.array().colwise() *=
        param.vector(format("h.{}.ln_1.weight", b)).array() * sqrt(n_embd);
    b_mlp1 += param.vector(format("h.{}.ln_2.bias", b)).transpose() * w_mlp1;
    w_mlp1.array().colwise() *=
        param.vector(format("h.{}.ln_2.weight", b)).array() * sqrt(n_embd);

    assert(w_attn1.rows() == n_embd);
    assert(w_attn1.cols() == 3 * n_embd);
    assert(w_attn2.rows() == n_embd);
    assert(w_attn2.cols() == n_embd);
    assert(w_mlp1.rows() == n_embd);
    assert(w_mlp1.cols() == 4 * n_embd);
    assert(w_mlp2.rows() == 4 * n_embd);
    assert(w_mlp2.cols() == n_embd);
    assert(b_attn1.size() == 3 * n_embd);
    assert(b_attn2.size() == n_embd);
    assert(b_mlp1.size() == 4 * n_embd);
    assert(b_mlp2.size() == n_embd);
  }

  /*
  def __call__(self, x):
      self.qkv = np.vstack(
          [self.qkv, normalise(x - np.mean(x)) @ self.w_attn1 + self.b_attn1]
      )
      attn = np.zeros(n_embd, dtype=x.dtype)
      for i in range(n_head):
          q = self.qkv[-1, D * i : D * (i + 1)]
          k = self.qkv[:, D * (n_head + i) : D * (n_head + i + 1)]
          v = self.qkv[:, D * (2 * n_head + i) : D * (2 * n_head + i + 1)]
          A = np.exp(q @ k.T / np.sqrt(D))
          attn[D * i : D * (i + 1)] = A @ v / np.sum(A)
      x += attn @ self.w_attn2 + self.b_attn2
      h = normalise(x - np.mean(x)) @ self.w_mlp1 + self.b_mlp1
      # h *= scipy.stats.norm.cdf(h)  # gelu
      h *= (1 + erf(h / np.sqrt(2))) / 2
      x += h @ self.w_mlp2 + self.b_mlp2
      return x
  */
  void operator()(VectorXf &x) {
    qkv.conservativeResize(qkv.rows() + 1, Eigen::NoChange);
    qkv.row(qkv.rows() - 1) =
        w_attn1.transpose() * (x.array() - x.mean()).matrix().normalized() +
        b_attn1;
    VectorXf attn(n_embd);
    for (int i = 0; i < n_head; i++) {
      VectorXf q = qkv.row(qkv.rows() - 1).segment(D * i, D);
      auto k = qkv.middleCols(D * (n_head + i), D);
      auto v = qkv.middleCols(D * (2 * n_head + i), D);
      VectorXf a = (k * q / sqrt(D)).array().exp();
      a /= a.array().sum();
      attn.segment(D * i, D) = v.transpose() * a;
    }
    x += w_attn2.transpose() * attn + b_attn2;
    VectorXf h =
        w_mlp1.transpose() * (x.array() - x.mean()).matrix().normalized() +
        b_mlp1;
    h.array() *= (1 + (h / sqrt(2)).array().erf()) / 2;
    x += w_mlp2.transpose() * h + b_mlp2;
  }
};
