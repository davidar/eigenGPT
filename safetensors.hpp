// https://gist.github.com/Narsil/5d6bf307995158ad2c4994f323967284

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace safetensors {
enum dtype_t {
  /// Boolean type
  kBOOL,
  /// Unsigned byte
  kUINT_8,
  /// Signed byte
  kINT_8,
  /// Signed integer (16-bit)
  kINT_16,
  /// Unsigned integer (16-bit)
  kUINT_16,
  /// Half-precision floating point
  kFLOAT_16,
  /// Brain floating point
  kBFLOAT_16,
  /// Signed integer (32-bit)
  kINT_32,
  /// Unsigned integer (32-bit)
  kUINT_32,
  /// Floating point (32-bit)
  kFLOAT_32,
  /// Floating point (64-bit)
  kFLOAT_64,
  /// Signed integer (64-bit)
  kINT_64,
  /// Unsigned integer (64-bit)
  kUINT_64,
};

NLOHMANN_JSON_SERIALIZE_ENUM(dtype_t, {
                                          {kBOOL, "BOOL"},
                                          {kUINT_8, "U8"},
                                          {kINT_8, "I8"},
                                          {kINT_16, "I16"},
                                          {kUINT_16, "U16"},
                                          {kFLOAT_16, "F16"},
                                          {kBFLOAT_16, "BF16"},
                                          {kINT_32, "I32"},
                                          {kUINT_32, "U32"},
                                          {kFLOAT_32, "F32"},
                                          {kFLOAT_64, "F64"},
                                          {kINT_64, "I64"},
                                          {kUINT_64, "U64"},
                                      })

struct metadata_t {
  dtype_t dtype;
  std::vector<size_t> shape;
  std::pair<size_t, size_t> data_offsets;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(metadata_t, dtype, shape, data_offsets)

class safetensors_t {
public:
  std::unordered_map<std::string, metadata_t> meta;
  std::vector<char> storage;

public:
  safetensors_t(std::basic_istream<char> &in) {
    uint64_t header_size = 0;

    // todo: handle exception
    in.read(reinterpret_cast<char *>(&header_size), sizeof header_size);

    std::vector<char> meta_block(header_size);
    in.read(meta_block.data(), static_cast<std::streamsize>(header_size));
    const auto metadatas = json::parse(meta_block);

    // How many bytes remaining to pre-allocate the storage tensor
    in.seekg(0, std::ios::end);
    std::streamsize f_size = in.tellg();
    in.seekg(8 + header_size, std::ios::beg);
    const auto tensors_size = f_size - 8 - header_size;

    storage.resize(tensors_size);

    // Read the remaining content
    in.read(storage.data(), static_cast<std::streamsize>(tensors_size));

    // Populate the meta lookup table
    if (metadatas.is_object()) {
      for (auto &[key, value] : metadatas.items()) {
        if (key != "__metadata__") {
          meta[key] = value.get<metadata_t>();
        }
      }
    }
  }

  inline size_t size() const { return meta.size(); }

  Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>
  matrix(std::string name) const {
    const metadata_t m = meta.at(name);
    assert(m.shape.size() == 2);
    float *data = (float *)&storage[m.data_offsets.first];
    return Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
        data, m.shape[0], m.shape[1]);
  }

  Eigen::Map<Eigen::VectorXf> vector(std::string name) const {
    const metadata_t m = meta.at(name);
    assert(m.shape.size() == 1);
    float *data = (float *)&storage[m.data_offsets.first];
    return Eigen::Map<Eigen::VectorXf>(data, m.shape[0]);
  }
};
} // namespace safetensors

#endif // SAFETENSORS_H
