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

typedef struct {
  float kv[n_ctx][2 * n_embd];
  int n_seq;
  float *w_attn1, *b_attn1, *w_attn2, *b_attn2, *w_mlp1, *b_mlp1, *w_mlp2,
      *b_mlp2, *w_ln1, *b_ln1, *w_ln2, *b_ln2;
} TransformerBlock;

TransformerBlock *newBlock(int b) {
  TransformerBlock *block = malloc(sizeof(TransformerBlock));
  block->n_seq = 0;
  block->w_attn1 = PARAM(block_offsets[b][0]);
  block->b_attn1 = PARAM(block_offsets[b][1]);
  block->w_attn2 = PARAM(block_offsets[b][2]);
  block->b_attn2 = PARAM(block_offsets[b][3]);
  block->w_ln1 = PARAM(block_offsets[b][4]);
  block->b_ln1 = PARAM(block_offsets[b][5]);
  block->w_mlp1 = PARAM(block_offsets[b][6]);
  block->b_mlp1 = PARAM(block_offsets[b][7]);
  block->w_mlp2 = PARAM(block_offsets[b][8]);
  block->b_mlp2 = PARAM(block_offsets[b][9]);
  block->w_ln2 = PARAM(block_offsets[b][10]);
  block->b_ln2 = PARAM(block_offsets[b][11]);
  return block;
}

void run(TransformerBlock *t, float x[n_embd]) {
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
    float *qkv = i < n_embd ? &q[i] : &t->kv[t->n_seq][i - n_embd];
    *qkv = t->b_attn1[i];
    FOR_EMBED(j, 1) {
      *qkv += t->w_attn1[j * (3 * n_embd) + i] *
              (t->b_ln1[j] + t->w_ln1[j] * norm_x[j] / sqrt(sqnorm / n_embd));
    }
  }
  t->n_seq++;

  // attention
  float attn[n_embd] = {0};
  float asum[n_head] = {0};
  for (int head = 0; head < n_head; head++) {
    for (int posn = 0; posn < t->n_seq; posn++) {
      float a = 0;
      for (int d = D * head; d < D + D * head; d++) {
        a += t->kv[posn][d] * q[d];
      }
      asum[head] += a = exp(a / sqrt(D));
      for (int d = D * head; d < D + D * head; d++) {
        attn[d] += t->kv[posn][d + n_embd] * a;
      }
    }
  }

  FOR_EMBED(i, 1) {
    float r = t->b_attn2[i];
    FOR_EMBED(j, 1) r += t->w_attn2[j * n_embd + i] * attn[j] / asum[j / D];
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
    h[i] = t->b_mlp1[i];
    FOR_EMBED(j, 1) {
      h[i] += t->w_mlp1[j * (4 * n_embd) + i] *
              (t->b_ln2[j] + t->w_ln2[j] * norm_x[j] / sqrt(sqnorm / n_embd));
    }
    h[i] *= (1 + erf(h[i] / sqrt(2))) / 2;
  }

  FOR_EMBED(i, 1) {
    FOR_EMBED(j, 4) x[i] += t->w_mlp2[j * n_embd + i] * h[j];
    x[i] += t->b_mlp2[i];
  }
}

float *wte, *wpe, *w_ln, *b_ln;
TransformerBlock *block[n_layer];

int predict(int token, int posn) {
  float x[n_embd];
  FOR_EMBED(i, 1) x[i] = wte[token * n_embd + i] + wpe[posn * n_embd + i];
  for (int i = 0; i < n_layer; i++) {
    run(block[i], x);
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
  storage = _binary_model_safetensors_start + 8 + header_size;
  wte = PARAM(wte_offset);
  wpe = PARAM(wpe_offset);
  w_ln = PARAM(w_ln_offset);
  b_ln = PARAM(b_ln_offset);
  for (int i = 0; i < n_layer; i++) {
    block[i] = newBlock(i);
  }

  int seed = time(NULL);
  printf("seed: %d\n", seed);
  srand(seed);

  int token = n_vocab - 1;
  for (int posn = 0; posn < n_ctx; posn++) {
    token = predict(token, posn);
    fprintf(stderr, "%s", vocab[token]);
  }
}
