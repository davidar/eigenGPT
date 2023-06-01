extern const unsigned char _binary_model_safetensors_start[];

#define header_size 14283

#define PARAM(offset) ((float *)(_binary_model_safetensors_start + 8 + header_size + offset))

int wte_offset = 287209472; // wte.weight
int wpe_offset = 267533312; // wpe.weight
int w_ln_offset = 9458688;  // ln_f.weight
int b_ln_offset = 85362688; // ln_f.bias

int block_offsets[12][12] = {
    {
        // layer 0
        270682112, // h.0.attn.c_attn.weight
        85365760,  // h.0.attn.c_attn.bias
        18432,     // h.0.attn.c_proj.weight
        9470976,   // h.0.attn.c_proj.bias
        451042304, // h.0.ln_1.weight
        242077696, // h.0.ln_1.bias
        115546112, // h.0.mlp.c_fc.weight
        18941952,  // h.0.mlp.c_fc.bias
        531575808, // h.0.mlp.c_proj.weight
        263332864, // h.0.mlp.c_proj.bias
        67518464,  // h.0.ln_2.weight
        206626816, // h.0.ln_2.bias
    },
    {
        // layer 1
        2380800,   // h.1.attn.c_attn.weight
        242056192, // h.1.attn.c_attn.bias
        513716224, // h.1.attn.c_proj.weight
        160421888, // h.1.attn.c_proj.bias
        172236800, // h.1.ln_1.weight
        213710848, // h.1.ln_1.bias
        232606720, // h.1.mlp.c_fc.weight
        277760000, // h.1.mlp.c_fc.bias
        141526016, // h.1.mlp.c_proj.weight
        94827520,  // h.1.mlp.c_proj.bias
        53871616,  // h.1.ln_2.weight
        94830592,  // h.1.ln_2.bias
    },
    {
        // layer 2
        541012992, // h.2.attn.c_attn.weight
        28397568,  // h.2.attn.c_attn.bias
        65159168,  // h.2.attn.c_proj.weight
        85347328,  // h.2.attn.c_proj.bias
        85350400,  // h.2.ln_1.weight
        451054592, // h.2.ln_1.bias
        451060736, // h.2.mlp.c_fc.weight
        3072,      // h.2.mlp.c_fc.bias
        500084736, // h.2.mlp.c_proj.weight
        451057664, // h.2.mlp.c_proj.bias
        176437248, // h.2.ln_2.weight
        223151104, // h.2.ln_2.bias
    },
    {
        // layer 3
        479375360, // h.3.attn.c_attn.weight
        18911232,  // h.3.attn.c_attn.bias
        176449536, // h.3.attn.c_proj.weight
        0,         // h.3.attn.c_proj.bias
        451048448, // h.3.ln_1.weight
        242074624, // h.3.ln_1.bias
        253892608, // h.3.mlp.c_fc.weight
        18920448,  // h.3.mlp.c_fc.bias
        223157248, // h.3.mlp.c_proj.weight
        15360,     // h.3.mlp.c_proj.bias
        524491776, // h.3.ln_2.weight
        202420224, // h.3.ln_2.bias
    },
    {
        // layer 4
        195342336, // h.4.attn.c_attn.weight
        150975488, // h.4.attn.c_attn.bias
        18954240,  // h.4.attn.c_proj.weight
        520291328, // h.4.attn.c_proj.bias
        206623744, // h.4.ln_1.weight
        150972416, // h.4.ln_1.bias
        185905152, // h.4.mlp.c_fc.weight
        53874688,  // h.4.mlp.c_fc.bias
        460500992, // h.4.mlp.c_proj.weight
        99027968,  // h.4.mlp.c_proj.bias
        242080768, // h.4.ln_2.weight
        53862400,  // h.4.ln_2.bias
    },
    {
        // layer 5
        524494848, // h.5.attn.c_attn.weight
        18932736,  // h.5.attn.c_attn.bias
        183542784, // h.5.attn.c_proj.weight
        169871360, // h.5.attn.c_proj.bias
        2377728,   // h.5.ln_1.weight
        40221696,  // h.5.ln_1.bias
        469938176, // h.5.mlp.c_fc.weight
        85374976,  // h.5.mlp.c_fc.bias
        242083840, // h.5.mlp.c_proj.weight
        21313536,  // h.5.mlp.c_proj.bias
        202423296, // h.5.ln_2.weight
        40218624,  // h.5.ln_2.bias
    },
    {
        // layer 6
        206629888, // h.6.attn.c_attn.weight
        242065408, // h.6.attn.c_attn.bias
        169877504, // h.6.attn.c_proj.weight
        263335936, // h.6.attn.c_proj.bias
        451045376, // h.6.ln_1.weight
        460497920, // h.6.ln_1.bias
        28406784,  // h.6.mlp.c_fc.weight
        232594432, // h.6.mlp.c_fc.bias
        85390336,  // h.6.mlp.c_proj.weight
        150963200, // h.6.mlp.c_proj.bias
        169862144, // h.6.ln_2.weight
        169868288, // h.6.ln_2.bias
    },
    {
        // layer 7
        132076544, // h.7.attn.c_attn.weight
        242043904, // h.7.attn.c_attn.bias
        251533312, // h.7.attn.c_proj.weight
        37843968,  // h.7.attn.c_proj.bias
        516075520, // h.7.ln_1.weight
        169874432, // h.7.ln_1.bias
        441605120, // h.7.mlp.c_fc.weight
        139154432, // h.7.mlp.c_fc.bias
        213713920, // h.7.mlp.c_proj.weight
        53865472,  // h.7.mlp.c_proj.bias
        85387264,  // h.7.ln_2.weight
        172239872, // h.7.ln_2.bias
    },
    {
        // layer 8
        53886976,  // h.8.attn.c_attn.weight
        251521024, // h.8.attn.c_attn.bias
        37859328,  // h.8.attn.c_proj.weight
        150966272, // h.8.attn.c_proj.bias
        251530240, // h.8.ln_1.weight
        242053120, // h.8.ln_1.bias
        490647552, // h.8.mlp.c_fc.weight
        181171200, // h.8.mlp.c_fc.bias
        9474048,   // h.8.mlp.c_proj.weight
        206620672, // h.8.mlp.c_proj.bias
        263329792, // h.8.ln_2.weight
        132061184, // h.8.ln_2.bias
    },
    {
        // layer 9
        21319680,  // h.9.attn.c_attn.weight
        9461760,   // h.9.attn.c_attn.bias
        181183488, // h.9.attn.c_proj.weight
        531572736, // h.9.attn.c_proj.bias
        441598976, // h.9.ln_1.weight
        185902080, // h.9.ln_1.bias
        277772288, // h.9.mlp.c_fc.weight
        37847040,  // h.9.mlp.c_fc.bias
        150984704, // h.9.mlp.c_proj.weight
        451051520, // h.9.mlp.c_proj.bias
        53859328,  // h.9.ln_2.weight
        169865216, // h.9.ln_2.bias
    },
    {
        // layer 10
        124983296, // h.10.attn.c_attn.weight
        176440320, // h.10.attn.c_attn.bias
        178808832, // h.10.attn.c_proj.weight
        181168128, // h.10.attn.c_proj.bias
        223154176, // h.10.ln_1.weight
        520288256, // h.10.ln_1.bias
        99031040,  // h.10.mlp.c_fc.weight
        520275968, // h.10.mlp.c_fc.bias
        71715840,  // h.10.mlp.c_proj.weight
        150969344, // h.10.mlp.c_proj.bias
        213707776, // h.10.ln_2.weight
        516078592, // h.10.ln_2.bias
    },
    {
        // layer 11
        108468224, // h.11.attn.c_attn.weight
        85353472,  // h.11.attn.c_attn.bias
        139166720, // h.11.attn.c_proj.weight
        21316608,  // h.11.attn.c_proj.bias
        441602048, // h.11.ln_1.weight
        53868544,  // h.11.ln_1.bias
        40227840,  // h.11.mlp.c_fc.weight
        132064256, // h.11.mlp.c_fc.bias
        160424960, // h.11.mlp.c_proj.weight
        520294400, // h.11.mlp.c_proj.bias
        40224768,  // h.11.ln_2.weight
        270679040, // h.11.ln_2.bias
    },
};
