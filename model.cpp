#include <math.h>
#include <sys/mman.h>

#include "blas.h"
#include "model.h"

Model::~Model() {
  delete[] h;
  if (mmap_data) {
    munmap(mmap_data, mmap_siz);
  }
}

void CausalSelfAttention::apply(const Tensorf<1> &out, const Tensorf<1> &xbuf,
                                int i, const Tensorf<2> &kvbuf) {
  const int emb_siz = 1600;
  const int num_heads = 25;

  // algebraic aggregators from the flash attention paper
  // https://arxiv.org/pdf/2205.14135.pdf section 3.1
  // but instead of combining blocks, I'm just reducing left-to-right in one
  // pass over the data
  Tensorf<1> flashatt_m(num_heads);  // maximum
  Tensorf<1> flashatt_l(num_heads);  // "l" is the denominator
  Tensorf<1> qbuf(emb_siz);
  Tensorf<1> ybuf(emb_siz);

  assert(xbuf.shape[0] == emb_siz);

  const int head_siz = 64;
  assert(emb_siz / num_heads == head_siz);

  float attn_scale = 1.0 / sqrt(head_siz);
  // first compute q, kv[i]

  {
    // matmul into q/kv; kv is cached for future invocations so write our
    // entry in there
    float *w = c_attn_weight.data;
    float *x = xbuf.data;
    float *b = c_attn_bias.data;
    float *q = qbuf.data;
    // matmul q = Qx
    for (int k = 0; k < emb_siz; k++) {
      *q++ = (*b++) + sdot(x, w, emb_siz);
      w += emb_siz;
    }
    // kv[i] = KVx
    float *kv = &kvbuf(i, 0);
    for (int k = 0; k < 2 * emb_siz; k++) {
      *kv++ = (*b++) + sdot(x, w, emb_siz);
      w += emb_siz;
    }
    {
      auto vbuf = kvbuf.slice(i);
      vbuf.data += emb_siz;
    }
  }

  // with all key-value entries populated, compute attention
  // the softmax is incrementally aggregated using the flash attention technique
  {
    float *qk = qbuf.data;
    float *kk = &kvbuf(0, 0);
    float *y = ybuf.data;
    // vk = kk + emb_siz
    memcpy(y, kk + emb_siz, emb_siz * sizeof(float));  // y is initially the first value for all heads
    for (int h = 0; h < num_heads; h++) {
      float a = sdot(qk, kk, head_siz) * attn_scale;
      flashatt_m[h] = a;
      flashatt_l[h] = 1;
      y += head_siz;
      qk += head_siz;
      kk += head_siz;
    }
    for (int j = 1; j <= i; j++) {
      float *qk = qbuf.data;
      float *kk = &kvbuf(j, 0);
      // vk = kk + emb_siz
      float *y = ybuf.data;
      for (int h = 0; h < num_heads; h++) {
        float a = sdot(qk, kk, head_siz) * attn_scale;
        if (a > flashatt_m[h]) {
          float e = expf(flashatt_m[h] - a); // <1.0
          // y = value + e*y
          sxpby(head_siz, kk + emb_siz, e, y);
          flashatt_l[h] = 1 + e*flashatt_l[h];
          flashatt_m[h] = a;
        } else {
          float e = expf(a - flashatt_m[h]); // <1.0
          saxpy(head_siz, e, kk + emb_siz, y);
          flashatt_l[h] += e;
        }
        y += head_siz;
        qk += head_siz;
        kk += head_siz;
      }
    }
    // scale y by 1/l
    y = ybuf.data;
    for (int h = 0; h < num_heads; h++) {
      float scale = 1.0 / flashatt_l[h];
      sscal(head_siz, scale, y);
      y += head_siz;
    }
  }

  // matmul the projection and sum into input
  // input += c_proj_weight @ ybuf + c_proj_bias
  {
    float *w = c_proj_weight.data;
    float *y = ybuf.data;
    float *o = out.data;
    for (int j = 0; j < emb_siz; j++) {
      *o++ += c_proj_bias[j] + sdot(y, w, emb_siz);
      w += emb_siz;
    }
  }
}

void LayerNorm::apply(Tensorf<1> &out, const Tensorf<1> &in) {
  float sum1 = 0;
  float sum2 = 0;
  float *i = in.data;
  int n = in.shape[0];
  for (int j = 0; j < n; j++) {
    float ij = *i++;
    sum1 += ij;
    sum2 += ij * ij;
  }
  // compute mean and variance
  float mean = sum1 / in.shape[0];
  float variance = sum2 / in.shape[0] - mean * mean;
  const float eps = 1e-5;  // layernorm default
  float invstddev = 1.0 / sqrt(variance + eps);
  float *w = weight.data;
  float *b = bias.data;
  float *o = out.data;
  i = in.data;
  // could vectorize, but not as performance critical as dot/matmuls
  for (int j = 0; j < n; j++) {
    *o++ = ((*i++) - mean) * invstddev * (*w++) + (*b++);
  }
}

void MLPBlock::apply(const Tensorf<1> &out, const Tensorf<1> &in) {
  int hidden_dim = 4*1600; 
  int emb_dim = 1600;

  assert(in.shape[0] == emb_dim);
  assert(c_fc_bias.shape[0] == hidden_dim);

  Tensorf<1> hbuf(hidden_dim);
  // fc part of block
  // input += mlp_c_proj_weight @ gelu(mlp_c_fc_weight @ xbuf + mlp_c_fc_bias) +
  // mlp_c_proj_bias
  {
    float *fc_w = c_fc_weight.data;
    float *x = in.data;
    float *h = hbuf.data;
    for (int j = 0; j < hidden_dim; j++) {
      float y = c_fc_bias[j] + sdot(x, fc_w, emb_dim);
      // float gelu = y * 0.5 * (1.0 + tanhf(0.7978845608028654 * (y + 0.044715 * y * y * y)));
      // use approximation xσ(1.702x), as expf is cheaper than tanhf
      float gelu = y / (1 + expf(-1.702*y));
      *h++ = gelu;
      fc_w += in.shape[0];
    }
  }
  // matmul the projection and sum into input
  {
    float *proj_w = c_proj_weight.data;
    float *o = out.data;
    float *h = hbuf.data;
    for (int j = 0; j < in.shape[0]; j++) {
      float sum = c_proj_bias[j] + sdot(h, proj_w, hidden_dim);
      *o++ += sum;
      proj_w += hidden_dim;
    }
  }
}

void Model::apply_lm_head(Tensorf<1> &emb_in, Tensorf<1> &logits) {
  assert(emb_in.shape[0] == embedding_dim);
  // layernorm and dot with embedding matrix
  ln_f.apply(emb_in, emb_in);
  const int ntokens = logits.shape[0];
  float *w = wte_weight.data;
  float m = -INFINITY;
  for (int j = 0; j < ntokens; j++) {
    logits[j] = sdot(emb_in.data, w, embedding_dim);
    if (logits[j] > m) {
      m = logits[j];
    }
    w += embedding_dim;
  }
  // subtract max for numerical stability
  for (int j = 0; j < ntokens; j++) {
    logits[j] -= m;
  }
}

void Model::apply_transformer(int token_id, int input_pos,
                              const Tensorf<3> &kvbuf,
                              const Tensorf<1> &emb_out) {
  for (int k = 0; k < embedding_dim; k++) {
    emb_out[k] = wte_weight(token_id, k) + wpe_weight(input_pos, k);
  }
  for (int layer = 0; layer < 48; layer++) {
    h[layer].apply(emb_out, input_pos, kvbuf.slice(layer));
  }
}

int sample_logits(float sampling_temperature, float uniform_sample, Tensorf<1> &logits) {
  // sample from logits (also normalizes logits to probabilities)
  int ntokens = logits.shape[0];
  float sum = 0;
  for (int j = 0; j < ntokens; j++) {
    logits[j] = expf(logits[j] / sampling_temperature);
    sum += logits[j];
  }
  for (int j = 0; j < ntokens; j++) {
    logits[j] /= sum;
  }
  float acc = 0;
  for (int j = 0; j < ntokens; j++) {
    acc += logits[j];
    if (acc >= uniform_sample) {
      return j;
    }
  }
  fprintf(stderr, "[sampling error? r=%f, acc=%f]\n", uniform_sample, acc);
  return 0;
}

float cross_entropy(const Tensorf<1> &logits, int index) {
  float sum = 0;
  // max has already been subtracted, so we just need the log sum
  for (int j = 0; j < logits.shape[0]; j++) {
    sum += expf(logits[j]);
  }
  return logits[index] - logf(sum);
}