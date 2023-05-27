#include <filesystem>
#include <fstream>
#include <iostream>

#include <Eigen/Dense>
#include <fmt/core.h>
#include <unsupported/Eigen/SpecialFunctions>

#include "safetensors.hpp"

using namespace Eigen;

using fmt::format;

namespace fs = std::filesystem;

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
    b_attn1 += param.vector(format("h.{}.ln_1.bias", b)).transpose() *
               param.matrix(format("h.{}.attn.c_attn.weight", b));
    w_attn1.array().colwise() *=
        param.vector(format("h.{}.ln_1.weight", b)).array() * sqrt(n_embd);
    b_mlp1 += param.vector(format("h.{}.ln_2.bias", b)).transpose() *
              param.matrix(format("h.{}.mlp.c_fc.weight", b));
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
  void operator()(Eigen::VectorXf &x) {
    qkv.conservativeResize(qkv.rows() + 1, Eigen::NoChange);
    VectorXf qkv_row =
        w_attn1.transpose() * (x.array() - x.mean()).matrix().normalized() +
        b_attn1;
    qkv.row(qkv.rows() - 1) = qkv_row;
    Eigen::VectorXf attn(n_embd);
    for (int i = 0; i < n_head; i++) {
      VectorXf q = qkv.row(qkv.rows() - 1).segment(D * i, D) / sqrt(D);
      MatrixXf k = qkv.middleCols(D * (n_head + i), D);
      MatrixXf v = qkv.middleCols(D * (2 * n_head + i), D);
      VectorXf qk = k * q;
      VectorXf a = qk.array().exp();
      attn.segment(D * i, D) = v.transpose() * a / a.array().sum();
    }
    x += w_attn2.transpose() * attn + b_attn2;
    Eigen::VectorXf h =
        w_mlp1.transpose() * (x.array() - x.mean()).matrix().normalized() +
        b_mlp1;
    h.array() *= (1 + (h / sqrt(2)).array().erf()) / 2;
    x += w_mlp2.transpose() * h + b_mlp2;
  }
};

int main() {
  fs::path path = "../gpt2/model.safetensors";
  if (!fs::exists(path))
    throw std::runtime_error("File does not exist");
  std::ifstream model(path, std::ios::binary);
  auto param = safetensors::safetensors_t(model);
  for (auto &[name, meta] : param.meta) {
    fmt::println("{}: {} {}", name, json(meta.dtype).dump(),
                 json(meta.shape).dump());
  }

  auto wte = param.matrix("wte.weight");
  auto wpe = param.matrix("wpe.weight");
  auto w_ln = param.vector("ln_f.weight") * sqrt(n_embd);
  auto b_ln = param.vector("ln_f.bias");

  std::vector<int> prompt = {36235, 39141, 18765, 1143, 326,
                             9061,  561,   530,   1110, 1716};
  int n_tokens_to_generate = 40;
  std::vector<int> tokens = prompt;
  int total = tokens.size() + n_tokens_to_generate;
  assert(total < n_ctx);
  std::vector<TransformerBlock> blocks;
  for (int b = 0; b < n_layer; b++) {
    blocks.emplace_back(param, b);
  }
  for (int posn = 0; posn < total; posn++) {
    int token = tokens[posn];
    Eigen::VectorXf x = wte.row(token) + wpe.row(posn);
    for (auto &block : blocks) {
      block(x);
    }
    Eigen::VectorXf final =
        (x.array() - x.mean()).matrix().normalized().array() * w_ln.array() +
        b_ln.array();
    Eigen::VectorXf logits = wte * final;
    if (posn + 1 >= tokens.size()) {
      int token;
      logits.maxCoeff(&token);
      tokens.push_back(token);
      fmt::println("{}", token);
    }
  }
}
