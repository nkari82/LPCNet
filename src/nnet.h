/* Copyright (c) 2018 Mozilla
   Copyright (c) 2017 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _NNET_H_
#define _NNET_H_

#define ACTIVATION_LINEAR  0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH    2
#define ACTIVATION_RELU    3
#define ACTIVATION_SOFTMAX 4

typedef struct {
  const float *bias;
  const float *input_weights;
  int nb_inputs;
  int nb_neurons;
  int activation;
} DenseLayer;

typedef struct {
  const float *bias;
  const float *input_weights;
  const float *factor;
  int nb_inputs;
  int nb_neurons;
  int nb_channels;
  int activation;
} MDenseLayer;

typedef struct {
  const float *bias;
  const float *input_weights;
  const float *recurrent_weights;
  int nb_inputs;
  int nb_neurons;
  int activation;
  int reset_after;
} GRULayer;

typedef struct {
  const float *bias;
  const float *diag_weights;
  const float *recurrent_weights;
  const int *idx;
  int nb_neurons;
  int activation;
  int reset_after;
} SparseGRULayer;

typedef struct {
  const float *bias;
  const float *input_weights;
  int nb_inputs;
  int kernel_size;
  int nb_neurons;
  int activation;
} Conv1DLayer;

typedef struct {
  const float *embedding_weights;
  int nb_inputs;
  int dim;
} EmbeddingLayer;

void compute_activation(float *output, float *input, int N, int activation);

void compute_dense(const DenseLayer *layer, float *output, const float *input);

void compute_mdense(const MDenseLayer *layer, float *output, const float *input);

void compute_gru(const GRULayer *gru, float *state, const float *input);

void compute_gru2(const GRULayer *gru, float *state, const float *input);

void compute_gru3(const GRULayer *gru, float *state, const float *input);

void compute_sparse_gru(const SparseGRULayer *gru, float *state, const float *input);

void compute_conv1d(const Conv1DLayer *layer, float *output, float *mem, const float *input);

void compute_embedding(const EmbeddingLayer *layer, float *output, int input);

void accum_embedding(const EmbeddingLayer *layer, float *output, int input);

int sample_from_pdf(const float *pdf, int N, float exp_boost, float pdf_floor);

void read_vector(void** vector, int elemt_size, FILE* f);

void read_embaded_layer(EmbeddingLayer* layer, FILE* f);

void read_dense_layer(DenseLayer* layer, FILE* f);

void read_conv1d_layer(Conv1DLayer* layer, FILE* f);

void read_gru_layer(GRULayer* layer, FILE* f);

#define GRU_A_EMBED_SIG_OUT_SIZE 1152
#define GRU_A_EMBED_PRED_OUT_SIZE 1152
#define GRU_A_EMBED_EXC_OUT_SIZE 1152
#define GRU_A_DENSE_FEATURE_OUT_SIZE 1152
#define EMBED_PITCH_OUT_SIZE 64
#define FEATURE_CONV1_OUT_SIZE 128
#define FEATURE_CONV1_STATE_SIZE (102*2)
#define FEATURE_CONV1_DELAY 1
#define FEATURE_CONV2_OUT_SIZE 128
#define FEATURE_CONV2_STATE_SIZE (128*2)
#define FEATURE_CONV2_DELAY 1
#define FEATURE_DENSE1_OUT_SIZE 128
#define EMBED_SIG_OUT_SIZE 128
#define FEATURE_DENSE2_OUT_SIZE 128
#define GRU_A_OUT_SIZE 384
#define GRU_A_STATE_SIZE 384
#define GRU_B_OUT_SIZE 16
#define GRU_B_STATE_SIZE 16
#define DUAL_FC_OUT_SIZE 256
#define SPARSE_GRU_A_OUT_SIZE 384
#define SPARSE_GRU_A_STATE_SIZE 384
#define MAX_RNN_NEURONS 384
#define MAX_CONV_INPUTS 384
#define MAX_MDENSE_TMP 512

typedef struct {
	EmbeddingLayer gru_a_embed_sig;
	EmbeddingLayer gru_a_embed_pred;
	EmbeddingLayer gru_a_embed_exc;
	DenseLayer gru_a_dense_feature;
	EmbeddingLayer embed_pitch;
	Conv1DLayer feature_conv1;
	Conv1DLayer feature_conv2;
	DenseLayer feature_dense1;
	EmbeddingLayer embed_sig;
	DenseLayer feature_dense2;
	GRULayer gru_a;
	GRULayer gru_b;
	MDenseLayer dual_fc;
	SparseGRULayer sparse_gru_a;
} NNetModel;

typedef struct {
	float feature_conv1_state[FEATURE_CONV1_STATE_SIZE];
	float feature_conv2_state[FEATURE_CONV2_STATE_SIZE];
	float gru_a_state[GRU_A_STATE_SIZE];
	float gru_b_state[GRU_B_STATE_SIZE];
} NNetState;

#endif /* _MLP_H_ */
