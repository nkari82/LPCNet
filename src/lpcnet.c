/* Copyright (c) 2018 Mozilla */
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdio.h>
#include "nnet.h"
#include "common.h"
#include "arch.h"
#include "lpcnet.h"
#include "lpcnet_private.h"

#define PREEMPH 0.85f

#define PITCH_GAIN_FEATURE 37
#define PDF_FLOOR 0.002

#define FRAME_INPUT_SIZE (NB_FEATURES + EMBED_PITCH_OUT_SIZE)


#if 0
static void print_vector(float *x, int N)
{
    int i;
    for (i=0;i<N;i++) printf("%f ", x[i]);
    printf("\n");
}
#endif

void run_frame_network(LPCNetState *lpcnet, float *condition, float *gru_a_condition, float *gru_b_condition, const float *features, int pitch)
{
    NNetState *net;
	NNetModel *model;
    float in[FRAME_INPUT_SIZE];
    float conv1_out[FEATURE_CONV1_OUT_SIZE];
    float conv2_out[FEATURE_CONV2_OUT_SIZE];
    float dense1_out[FEATURE_DENSE1_OUT_SIZE];
    net = &lpcnet->nnet;
	model = lpcnet->model;
    RNN_COPY(in, features, NB_FEATURES);
    compute_embedding(&model->embed_pitch, &in[NB_FEATURES], pitch);
    compute_conv1d(&model->feature_conv1, conv1_out, net->feature_conv1_state, in);
    if (lpcnet->frame_count < FEATURE_CONV1_DELAY) RNN_CLEAR(conv1_out, FEATURE_CONV1_OUT_SIZE);
    compute_conv1d(&model->feature_conv2, conv2_out, net->feature_conv2_state, conv1_out);
    celt_assert(FRAME_INPUT_SIZE == FEATURE_CONV2_OUT_SIZE);
    if (lpcnet->frame_count < FEATURES_DELAY) RNN_CLEAR(conv2_out, FEATURE_CONV2_OUT_SIZE);
    memmove(lpcnet->old_input[1], lpcnet->old_input[0], (FEATURES_DELAY-1)*FRAME_INPUT_SIZE*sizeof(in[0]));
    memcpy(lpcnet->old_input[0], in, FRAME_INPUT_SIZE*sizeof(in[0]));
    compute_dense(&model->feature_dense1, dense1_out, conv2_out);
    compute_dense(&model->feature_dense2, condition, dense1_out);
    compute_dense(&model->gru_a_dense_feature, gru_a_condition, condition);
	compute_dense(&model->gru_b_dense_feature, gru_b_condition, condition);
    if (lpcnet->frame_count < 1000) lpcnet->frame_count++;
}

void run_sample_network(NNetState *net, NNetModel *model, float *pdf, const float *condition, const float *gru_a_condition, const float *gru_b_condition, int last_exc, int last_sig, int pred)
{
    float gru_a_input[3*GRU_A_STATE_SIZE];
    float gru_b_input[3*GRU_B_STATE_SIZE];
    float in_b[GRU_A_STATE_SIZE];
    RNN_COPY(gru_a_input, gru_a_condition, 3*GRU_A_STATE_SIZE);
    accum_embedding(&model->gru_a_embed_sig, gru_a_input, last_sig);
    accum_embedding(&model->gru_a_embed_pred, gru_a_input, pred);
    accum_embedding(&model->gru_a_embed_exc, gru_a_input, last_exc);
    /*compute_gru3(&gru_a, net->gru_a_state, gru_a_input);*/
    compute_sparse_gru(&model->sparse_gru_a, net->gru_a_state, gru_a_input);
    RNN_COPY(in_b, net->gru_a_state, GRU_A_STATE_SIZE);
    RNN_COPY(gru_b_input, gru_b_condition, 3*GRU_B_STATE_SIZE);
	compute_gruB2(&model->gru_b, gru_b_input, net->gru_b_state, in_b);
    compute_mdense(&model->dual_fc, pdf, net->gru_b_state);
}

LPCNET_EXPORT int lpcnet_get_size()
{
    return sizeof(LPCNetState);
}

LPCNET_EXPORT int lpcnet_init(LPCNetState *lpcnet)
{
    memset(lpcnet, 0, offsetof(struct LPCNetState, model));
    lpcnet->last_exc = lin2ulaw(0.f);
    return 0;
}


LPCNET_EXPORT LPCNetState *lpcnet_create()
{
    LPCNetState *lpcnet;
    lpcnet = (LPCNetState *)calloc(lpcnet_get_size(), 1);
	lpcnet->model = NULL;
    lpcnet_init(lpcnet);
    return lpcnet;
}

LPCNET_EXPORT void lpcnet_destroy(LPCNetState *lpcnet)
{
	lpcnet_unload(lpcnet);
    free(lpcnet);
}

LPCNET_EXPORT int lpcnet_synthesize(LPCNetState *lpcnet, const float *features, short *output, int N)
{
	NNetState* net;
	NNetModel* model;
    int i;
    float condition[FEATURE_DENSE2_OUT_SIZE];
    float lpc[LPC_ORDER];
    float pdf[DUAL_FC_OUT_SIZE];
    float gru_a_condition[3*GRU_A_STATE_SIZE];
    float gru_b_condition[3*GRU_B_STATE_SIZE];
    int pitch;
    float pitch_gain;
	net = &lpcnet->nnet;
	model = lpcnet->model;
	if (!model) return -1;
    /* Matches the Python code -- the 0.1 avoids rounding issues. */
    pitch = (int)floor(.1 + 50*features[36]+100);
    pitch_gain = lpcnet->old_gain[FEATURES_DELAY-1];
    memmove(&lpcnet->old_gain[1], &lpcnet->old_gain[0], (FEATURES_DELAY-1)*sizeof(lpcnet->old_gain[0]));
    lpcnet->old_gain[0] = features[PITCH_GAIN_FEATURE];
    run_frame_network(lpcnet, condition, gru_a_condition, gru_b_condition, features, pitch);
    memcpy(lpc, lpcnet->old_lpc[FEATURES_DELAY-1], LPC_ORDER*sizeof(lpc[0]));
    memmove(lpcnet->old_lpc[1], lpcnet->old_lpc[0], (FEATURES_DELAY-1)*LPC_ORDER*sizeof(lpc[0]));
    lpc_from_cepstrum(lpcnet->old_lpc[0], features);
    if (lpcnet->frame_count <= FEATURES_DELAY)
    {
        RNN_CLEAR(output, N);
        return 1;
    }
    for (i=0;i<N;i++)
    {
        int j;
        float pcm;
        int exc;
        int last_sig_ulaw;
        int pred_ulaw;
        float pred = 0;
        for (j=0;j<LPC_ORDER;j++) pred -= lpcnet->last_sig[j]*lpc[j];
        last_sig_ulaw = lin2ulaw(lpcnet->last_sig[0]);
        pred_ulaw = lin2ulaw(pred);
        run_sample_network(net, model, pdf, condition, gru_a_condition, gru_b_condition, lpcnet->last_exc, last_sig_ulaw, pred_ulaw);
        exc = sample_from_pdf(pdf, DUAL_FC_OUT_SIZE, MAX16(0, 1.5f*pitch_gain - .5f), PDF_FLOOR);
        pcm = pred + ulaw2lin(exc);
        RNN_MOVE(&lpcnet->last_sig[1], &lpcnet->last_sig[0], LPC_ORDER-1);
        lpcnet->last_sig[0] = pcm;
        lpcnet->last_exc = exc;
        pcm += PREEMPH*lpcnet->deemph_mem;
        lpcnet->deemph_mem = pcm;
        if (pcm<-32767) pcm = -32767;
        if (pcm>32767) pcm = 32767;
        output[i] = (int)floor(.5 + pcm);
    }
    return 0;
}


LPCNET_EXPORT int lpcnet_decoder_get_size()
{
  return sizeof(LPCNetDecState);
}

LPCNET_EXPORT int lpcnet_decoder_init(LPCNetDecState *st)
{
  memset(st, 0, lpcnet_decoder_get_size());
  lpcnet_init(&st->lpcnet_state);
  return 0;
}

LPCNET_EXPORT LPCNetDecState *lpcnet_decoder_create()
{
  LPCNetDecState *st;
  st = malloc(lpcnet_decoder_get_size());
  lpcnet_decoder_init(st);
  return st;
}

LPCNET_EXPORT void lpcnet_decoder_destroy(LPCNetDecState *st)
{
  free(st);
}

LPCNET_EXPORT int lpcnet_decode(LPCNetDecState *st, const unsigned char *buf, short *pcm)
{
  int k;
  float features[4][NB_TOTAL_FEATURES];
  decode_packet(features, st->vq_mem, buf);
  for (k=0;k<4;k++) {
    lpcnet_synthesize(&st->lpcnet_state, features[k], &pcm[k*FRAME_SIZE], FRAME_SIZE);
  }
  return 0;
}

LPCNET_EXPORT int lpcnet_load(LPCNetState *lpcnet, const char* path)
{
	NNetModel* model = lpcnet->model;
	FILE* file = fopen(path, "rb");
	if (!file) return -1;

	lpcnet_unload(lpcnet);

	model = (NNetModel*)calloc(sizeof(NNetModel), 1);

	read_embaded_layer(&model->gru_a_embed_sig, file);
	read_embaded_layer(&model->gru_a_embed_pred, file);
	read_embaded_layer(&model->gru_a_embed_exc, file);
	read_dense_layer(&model->gru_a_dense_feature, file);
    read_dense_layer(&model->gru_b_dense_feature, file);
	read_embaded_layer(&model->embed_pitch, file);
	read_conv1d_layer(&model->feature_conv1, file);
	read_conv1d_layer(&model->feature_conv2, file);
	read_dense_layer(&model->feature_dense1, file);
	read_embaded_layer(&model->embed_sig, file);
	read_dense_layer(&model->feature_dense2, file);
	read_gru_layer(&model->gru_a, file);
	read_gru_layer(&model->gru_b, file);
	
	read_vector((void**)&model->dual_fc.input_weights, sizeof(float), file);
	read_vector((void**)&model->dual_fc.bias, sizeof(float), file);
	read_vector((void**)&model->dual_fc.factor, sizeof(float), file);
	fread(&model->dual_fc.nb_inputs, sizeof(int), 1, file);
	fread(&model->dual_fc.nb_neurons, sizeof(int), 1, file);
	fread(&model->dual_fc.nb_channels, sizeof(int), 1, file);
	fread(&model->dual_fc.activation, sizeof(int), 1, file);

	read_vector((void**)&model->sparse_gru_a.diag_weights, sizeof(float), file);
	read_vector((void**)&model->sparse_gru_a.recurrent_weights, sizeof(float), file);
	read_vector((void**)&model->sparse_gru_a.idx, sizeof(int), file);
	read_vector((void**)&model->sparse_gru_a.bias, sizeof(float), file);
	fread(&model->sparse_gru_a.nb_neurons, sizeof(int), 1, file);
	fread(&model->sparse_gru_a.activation, sizeof(int), 1, file);
	fread(&model->sparse_gru_a.reset_after, sizeof(int), 1, file);
    lpcnet->model = model;
	return 0;
}

LPCNET_EXPORT void lpcnet_unload(LPCNetState *lpcnet)
{
	NNetModel* model = lpcnet->model;
	if (!model) return;

	free((void*)model->gru_a_embed_sig.embedding_weights);
	free((void*)model->gru_a_embed_pred.embedding_weights);
	free((void*)model->gru_a_embed_exc.embedding_weights);
	free((void*)model->gru_a_dense_feature.input_weights);
	free((void*)model->gru_a_dense_feature.bias);
	free((void*)model->gru_b_dense_feature.input_weights);
	free((void*)model->gru_b_dense_feature.bias);
	free((void*)model->embed_pitch.embedding_weights);
	free((void*)model->feature_conv1.bias);
	free((void*)model->feature_conv1.input_weights);
	free((void*)model->feature_conv2.bias);
	free((void*)model->feature_conv2.input_weights);
	free((void*)model->feature_dense1.bias);
	free((void*)model->feature_dense1.input_weights);
	free((void*)model->embed_sig.embedding_weights);
	free((void*)model->feature_dense2.bias);
	free((void*)model->feature_dense2.input_weights);
	free((void*)model->gru_a.bias);
	free((void*)model->gru_a.input_weights);
	free((void*)model->gru_a.recurrent_weights);
	free((void*)model->gru_b.bias);
	free((void*)model->gru_b.input_weights);
	free((void*)model->gru_b.recurrent_weights);
	free((void*)model->dual_fc.bias);
	free((void*)model->dual_fc.input_weights);
	free((void*)model->dual_fc.factor);
	free((void*)model->sparse_gru_a.bias);
	free((void*)model->sparse_gru_a.diag_weights);
	free((void*)model->sparse_gru_a.idx);
	free((void*)model->sparse_gru_a.recurrent_weights);

	free(model);
	lpcnet->model = NULL;
}




