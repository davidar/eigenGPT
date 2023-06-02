#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "model_offsets.h"
#include "vocab.h"

extern const float MODEL[];

#define n_ctx 1024
#define n_embd 768
#define n_head 12
#define n_layer 12
#define D (n_embd / n_head)

#define FOR_EMBED(var, mul) for (int var = 0; var < mul * n_embd; var++)

float kv[n_layer][n_ctx][2 * n_embd];

int main() {
  int seed = time(NULL);
  printf("seed: %d\n", seed);
  srand(seed);

  int token = n_vocab - 1;
  for (int n_seq = 0; n_seq < n_ctx; n_seq++) {
    float x[n_embd];
    FOR_EMBED(i, 1) {
      x[i] = MODEL[wte_offset + token * n_embd + i] +
             MODEL[wpe_offset + n_seq * n_embd + i];
    }
    for (int b = 0; b < n_layer; b++) {
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
        *qkv = MODEL[b_attn1_offset[b] + i];
        FOR_EMBED(j, 1) {
          *qkv +=
              MODEL[w_attn1_offset[b] + j * (3 * n_embd) + i] *
              (MODEL[b_ln1_offset[b] + j] +
               MODEL[w_ln1_offset[b] + j] * norm_x[j] / sqrt(sqnorm / n_embd));
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
        float r = MODEL[b_attn2_offset[b] + i];
        FOR_EMBED(j, 1) {
          r +=
              MODEL[w_attn2_offset[b] + j * n_embd + i] * attn[j] / asum[j / D];
        }
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
        h[i] = MODEL[b_mlp1_offset[b] + i];
        FOR_EMBED(j, 1) {
          h[i] +=
              MODEL[w_mlp1_offset[b] + j * (4 * n_embd) + i] *
              (MODEL[b_ln2_offset[b] + j] +
               MODEL[w_ln2_offset[b] + j] * norm_x[j] / sqrt(sqnorm / n_embd));
        }
        h[i] *= (1 + erf(h[i] / sqrt(2))) / 2;
      }

      FOR_EMBED(i, 1) {
        FOR_EMBED(j, 4) x[i] += MODEL[w_mlp2_offset[b] + j * n_embd + i] * h[j];
        x[i] += MODEL[b_mlp2_offset[b] + i];
      }
    }

    // final layer norm
    float sum = 0, sqnorm = 0;
    FOR_EMBED(i, 1) sum += x[i];
    FOR_EMBED(i, 1) {
      x[i] -= sum / n_embd;
      sqnorm += x[i] * x[i];
    }
    FOR_EMBED(i, 1) {
      x[i] = MODEL[b_ln_offset + i] +
             MODEL[w_ln_offset + i] * x[i] / sqrt(sqnorm / n_embd);
    }

    float logit[n_vocab] = {0};
    float max_logit;
    for (int t = 0; t < n_vocab; t++) {
      FOR_EMBED(i, 1)
      logit[t] += MODEL[wte_offset + t * n_embd + i] * x[i];
      if (t == 0 || logit[t] > max_logit) {
        max_logit = logit[t];
      }
    }

    // softmax with temperature
    float sum_exp = 0;
    for (int t = 0; t < n_vocab; t++) {
      logit[t] = exp((logit[t] - max_logit) / 0.7);
      sum_exp += logit[t];
    }

    // random sampling
    float r = (float)rand() / RAND_MAX;
    token = n_vocab - 1;
    for (int t = 0; t < n_vocab; t++) {
      r -= logit[t] / sum_exp;
      if (r <= 0) {
        token = t;
        break;
      }
    }
    if (token == n_vocab - 1) {
      break;
    }
    fprintf(stderr, "%s", vocab[token]);
  }
}
