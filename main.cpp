#include <filesystem>
#include <fstream>
#include <iostream>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>

#include "safetensors.hpp"

namespace fs = std::filesystem;

using Eigen::MatrixXd;

int main() {
  MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;

  fs::path path = "../model.safetensors";
  if (!fs::exists(path))
    throw std::runtime_error("File does not exist");
  std::ifstream model(path, std::ios::binary);
  auto tensors = huggingface::safetensors::deserialize(model);
  for (auto &[name, meta] : tensors.meta) {
    spdlog::info("{}: {} {}", name, json(meta.dtype).dump(), json(meta.shape).dump());
  }
  float *b_ln = (float*) tensors["ln_f.bias"].data();
  for (int i = 0; i < 768; i++) {
    std::cout << b_ln[i] << ",";
  }
}
