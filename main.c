#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "model_offsets.h"
#include "vocab.h"

#define n_ctx 1024
#define n_embd 768
#define n_head 12
#define n_layer 12
#define D (n_embd / n_head)

#define FOR_EMBED(var, mul) for (int var = 0; var < mul * n_embd; var++)

#define w_attn1(b) PARAM(block_offsets[b][0])
#define b_attn1(b) PARAM(block_offsets[b][1])
#define w_attn2(b) PARAM(block_offsets[b][2])
#define b_attn2(b) PARAM(block_offsets[b][3])
#define w_ln1(b) PARAM(block_offsets[b][4])
#define b_ln1(b) PARAM(block_offsets[b][5])
#define w_mlp1(b) PARAM(block_offsets[b][6])
#define b_mlp1(b) PARAM(block_offsets[b][7])
#define w_mlp2(b) PARAM(block_offsets[b][8])
#define b_mlp2(b) PARAM(block_offsets[b][9])
#define w_ln2(b) PARAM(block_offsets[b][10])
#define b_ln2(b) PARAM(block_offsets[b][11])

#define wte PARAM(wte_offset)
#define wpe PARAM(wpe_offset)
#define w_ln PARAM(w_ln_offset)
#define b_ln PARAM(b_ln_offset)

int n_seq = 0;
float kv[n_layer][n_ctx][2 * n_embd];

void block(int b, float x[n_embd]) {
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
    float *qkv = i < n_embd ? &q[i] : &kv[b][n_seq][i - n_embd];
    *qkv = b_attn1(b)[i];
    FOR_EMBED(j, 1) {
      *qkv += w_attn1(b)[j * (3 * n_embd) + i] *
              (b_ln1(b)[j] + w_ln1(b)[j] * norm_x[j] / sqrt(sqnorm / n_embd));
    }
  }

  // attention
  float attn[n_embd] = {0};
  float asum[n_head] = {0};
  for (int head = 0; head < n_head; head++) {
    for (int posn = 0; posn < n_seq + 1; posn++) {
      float a = 0;
      for (int d = D * head; d < D + D * head; d++) {
        a += kv[b][posn][d] * q[d];
      }
      asum[head] += a = exp(a / sqrt(D));
      for (int d = D * head; d < D + D * head; d++) {
        attn[d] += kv[b][posn][d + n_embd] * a;
      }
    }
  }

  FOR_EMBED(i, 1) {
    float r = b_attn2(b)[i];
    FOR_EMBED(j, 1) r += w_attn2(b)[j * n_embd + i] * attn[j] / asum[j / D];
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
    h[i] = b_mlp1(b)[i];
    FOR_EMBED(j, 1) {
      h[i] += w_mlp1(b)[j * (4 * n_embd) + i] *
              (b_ln2(b)[j] + w_ln2(b)[j] * norm_x[j] / sqrt(sqnorm / n_embd));
    }
    h[i] *= (1 + erf(h[i] / sqrt(2))) / 2;
  }

  FOR_EMBED(i, 1) {
    FOR_EMBED(j, 4) x[i] += w_mlp2(b)[j * n_embd + i] * h[j];
    x[i] += b_mlp2(b)[i];
  }
}

int predict(int token, int posn) {
  float x[n_embd];
  FOR_EMBED(i, 1) x[i] = wte[token * n_embd + i] + wpe[posn * n_embd + i];
  for (int i = 0; i < n_layer; i++) {
    block(i, x);
  }

  float sum = 0, sqnorm = 0;
  FOR_EMBED(i, 1) sum += x[i];
  FOR_EMBED(i, 1) {
    x[i] -= sum / n_embd;
    sqnorm += x[i] * x[i];
  }
  FOR_EMBED(i, 1) x[i] = b_ln[i] + w_ln[i] * x[i] / sqrt(sqnorm / n_embd);

  float logit[n_vocab] = {0};
  float max_logit;
  for (int token = 0; token < n_vocab; token++) {
    FOR_EMBED(i, 1) logit[token] += wte[token * n_embd + i] * x[i];
    if (token == 0 || logit[token] > max_logit) {
      max_logit = logit[token];
    }
  }

  float sum_exp = 0;
  for (int token = 0; token < n_vocab; token++) {
    logit[token] = exp((logit[token] - max_logit) / 0.7);
    sum_exp += logit[token];
  }
  float r = (float)rand() / RAND_MAX;
  for (int token = 0; token < n_vocab; token++) {
    r -= logit[token] / sum_exp;
    if (r <= 0) {
      return token;
    }
  }
  return n_vocab - 1;
}

int main() {
  int seed = time(NULL);
  printf("seed: %d\n", seed);
  srand(seed);

  int token = n_vocab - 1;
  for (int posn = 0; posn < n_ctx; posn++) {
    n_seq = posn;
    token = predict(token, posn);
    if (token == n_vocab - 1) {
      break;
    }
    fprintf(stderr, "%s", vocab[token]);
  }
}
