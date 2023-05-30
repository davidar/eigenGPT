// https://gist.github.com/Narsil/5d6bf307995158ad2c4994f323967284

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// http://gareus.org/wiki/embedding_resources_in_executables
extern const unsigned char _binary_model_safetensors_start[];

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
  const unsigned char *storage;

public:
  safetensors_t() {
    uint64_t header_size =
        *reinterpret_cast<const uint64_t *>(_binary_model_safetensors_start);

    std::vector<char> meta_block(header_size);
    memcpy(meta_block.data(),
           _binary_model_safetensors_start + sizeof header_size,
           static_cast<size_t>(header_size));
    const auto metadatas = json::parse(meta_block);

    storage =
        _binary_model_safetensors_start + sizeof header_size + header_size;

    // std::cerr << "header_size: " << header_size << std::endl;

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

  float *data(std::string name) const {
    return data(meta.at(name).data_offsets.first);
  }
  float *data(size_t offset) const { return (float *)&storage[offset]; }
};
} // namespace safetensors

#endif // SAFETENSORS_H
