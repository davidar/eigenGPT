#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "model_offsets.h"
#include "vocab.h"

extern const float M[];

#define D 64
#define E 768
#define F(v, m) for (int v = 0; v < m * E; v++)
#define G F(i, 1)

float kv[12][1024][2 * E];

int main() {
  srand(time(0));
  int T = V;
  for (int n = 0; n < 1024; n++) {
    float x[E];
    G x[i] = M[wt + T * E + i] + M[wp + n * E + i];
    for (int b = 0; b < 12; b++) {
      float y[E];
      float s = 0, sq = 0;
      G s += x[i];
      G y[i] = x[i] - s / E, sq += y[i] * y[i];

      // update kv cache
      float q[E];
      F(j, 3) {
        float *qkv = j < E ? &q[j] : &kv[b][n][j - E];
        *qkv = M[b1[b] + j];
        G *qkv += M[w1[b] + i * 3 * E + j] * (M[c1[b] + i] + M[g1[b] + i] * y[i] / sqrt(sq / E));
      }

      // attention
      float attn[E] = {0};
      float asum[12] = {0};
      for (int h = 0; h < 12; h++) {
        for (int p = 0; p < n + 1; p++) {
          float a = 0;
          for (int d = D * h; d < D + D * h; d++) a += kv[b][p][d] * q[d];
          asum[h] += a = exp(a / sqrt(D));
          for (int d = D * h; d < D + D * h; d++) attn[d] += kv[b][p][d + E] * a;
        }
      }

      G {
        float r = M[b2[b] + i];
        F(j, 1) r += M[w2[b] + j * E + i] * attn[j] / asum[j / D];
        x[i] += r;
        s += r;
      }

      sq = 0;
      G y[i] = x[i] - s / E, sq += y[i] * y[i];

      // mlp
      float h[4 * E];
      F(j, 4) {
        h[j] = M[b3[b] + j];
        G h[j] += M[w3[b] + i * 4 * E + j] * (M[c2[b] + i] + M[g2[b] + i] * y[i] / sqrt(sq / E));
        h[j] *= (1 + erf(h[j] / sqrt(2))) / 2;
      }

      G {
        F(j, 4) x[i] += M[w4[b] + j * E + i] * h[j];
        x[i] += M[b4[b] + i];
      }
    }

    // final layer norm
    float s = 0, sq = 0;
    G s += x[i];
    G x[i] -= s / E, sq += x[i] * x[i];
    G x[i] = M[c0 + i] + M[g0 + i] * x[i] / sqrt(sq / E);

    float logit[V + 1] = {0};
    float max_logit;
    for (int t = 0; t <= V; t++) {
      G logit[t] += M[wt + t * E + i] * x[i];
      if (t == 0 || logit[t] > max_logit) {
        max_logit = logit[t];
      }
    }

    // softmax with temperature
    float sum_exp = 0;
    for (int t = 0; t <= V; t++) {
      logit[t] = exp((logit[t] - max_logit) / 0.7);
      sum_exp += logit[t];
    }

    // random sampling
    float r = (float)rand() / RAND_MAX;
    T = V;
    for (int t = 0; t <= V; t++) {
      r -= logit[t] / sum_exp;
      if (r <= 0) {
        T = t;
        break;
      }
    }
    if (T == V) break;
    fprintf(stderr, "%s", v[T]);
  }
}
