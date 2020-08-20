/* Copyright (c) 2017-2018 Mozilla */
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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <sox.h>
#if defined(__cplusplus)
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <regex>
#include <type_traits>

#ifdef __has_include
#  if __has_include(<filesystem>)
#    include <filesystem>
	  namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
	  namespace fs = std::experimental::filesystem;
#  else
	  static_assert(false, "must include filesystem!");
#  endif
#endif

#define CPPGLOB_STATIC
#include <cppglob/glob.hpp>  // cppglob::glob
#include <cppglob/iglob.hpp>  // cppglob::iglob

//#define CXXOPTS_NO_EXCEPTIONS
#include <cxxopts.hpp>

extern "C"
{
#endif
#include "kiss_fft.h"
#include "common.h"
#include "freq.h"
#include "pitch.h"
#include "arch.h"
#include "celt_lpc.h"
#include "lpcnet.h"
#include "lpcnet_private.h"
#if defined(__cplusplus)
}
#endif

static void biquad(float* y, float mem[2], const float* x, const float* b, const float* a, int N) {
    int i;
    for (i = 0; i < N; i++) {
        float xi, yi;
        xi = x[i];
        yi = x[i] + mem[0];
        mem[0] = mem[1] + (b[0] * (double)xi - a[0] * (double)yi);
        mem[1] = (b[1] * (double)xi - a[1] * (double)yi);
        y[i] = yi;
    }
}

static float uni_rand() {
    return rand() / (double)RAND_MAX - .5;
}

static void rand_resp(float* a, float* b) {
    a[0] = .75 * uni_rand();
    a[1] = .75 * uni_rand();
    b[0] = .75 * uni_rand();
    b[1] = .75 * uni_rand();
}

void compute_noise(int* noise, float noise_std) {
    int i;
    for (i = 0; i < FRAME_SIZE; i++) {
        noise[i] = (int)floor(.5 + noise_std * .707 * (log_approx((float)rand() / RAND_MAX) - log_approx((float)rand() / RAND_MAX)));
    }
}


void write_audio(LPCNetEncState* st, const short* pcm, const int* noise, FILE* file) {
    int i, k;
    for (k = 0; k < 4; k++) {
        unsigned char data[4 * FRAME_SIZE];
        for (i = 0; i < FRAME_SIZE; i++) {
            float p = 0;
            float e;
            int j;
            for (j = 0; j < LPC_ORDER; j++) p -= st->features[k][2 * NB_BANDS + 3 + j] * st->sig_mem[j];
            e = lin2ulaw(pcm[k * FRAME_SIZE + i] - p);
            /* Signal. */
            data[4 * i] = lin2ulaw(st->sig_mem[0]);
            /* Prediction. */
            data[4 * i + 1] = lin2ulaw(p);
            /* Excitation in. */
            data[4 * i + 2] = st->exc_mem;
            /* Excitation out. */
            data[4 * i + 3] = e;
            /* Simulate error on excitation. */
            e += noise[k * FRAME_SIZE + i];
            e = IMIN(255, IMAX(0, e));

            RNN_MOVE(&st->sig_mem[1], &st->sig_mem[0], LPC_ORDER - 1);
            st->sig_mem[0] = p + ulaw2lin(e);
            st->exc_mem = e;
        }
        fwrite(data, 4 * FRAME_SIZE, 1, file);
    }
}

static short float2short(float x)
{
    int i;
    i = (int)floor(.5 + x);
    return IMAX(-32767, IMIN(32767, i));
}

static void copy(FILE* from, FILE* to)
{
	char buffer[2048];
	size_t sz(0);
	while (!feof(from))
	{
		sz = fread(buffer, 1, sizeof(buffer), from);
		fwrite(buffer, 1, sz, to);
	}
}

static void convert_to(const fs::path& in_path, const fs::path& out_path, const char* type = "sw")
{
	fprintf(stdout, "Convert: %s\n", in_path.string().c_str());

	sox_signalinfo_t interm_signal;
	sox_encodinginfo_t out_encoding = { SOX_ENCODING_SIGN2, 16, 0, sox_option_default, sox_option_default, sox_option_default, sox_false };
	sox_signalinfo_t out_signal = { 16000, 1, 16, 0, NULL };

	sox_signalinfo_t default_in_signal = { 16000, 1, 16, 0, NULL };
	sox_signalinfo_t* in_signal = NULL;

	if (strcmp(type, "wav") == 0)
		in_signal = &default_in_signal;

	char* args[10];
	sox_format_t* in = sox_open_read(in_path.string().c_str(), in_signal, NULL, NULL);
	sox_format_t* out = sox_open_write(out_path.string().c_str(), &out_signal, &out_encoding, type, NULL, NULL);

	sox_effects_chain_t* chain = sox_create_effects_chain(&in->encoding, &out->encoding);
	interm_signal = in->signal; /* NB: deep copy */

	sox_effect_t* e = sox_create_effect(sox_find_effect("input"));
	args[0] = (char *)in, sox_effect_options(e, 1, args);
	sox_add_effect(chain, e, &interm_signal, &in->signal);
	free(e);

	if (in->signal.rate != out->signal.rate)
	{
		e = sox_create_effect(sox_find_effect("rate"));
		sox_effect_options(e, 0, NULL);
		sox_add_effect(chain, e, &interm_signal, &out->signal);
		free(e);
	}

	if (in->signal.channels != out->signal.channels)
	{
		e = sox_create_effect(sox_find_effect("channels"));
		sox_effect_options(e, 0, NULL);
		sox_add_effect(chain, e, &interm_signal, &out->signal);
		free(e);
	}

	e = sox_create_effect(sox_find_effect("output"));
	args[0] = (char *)out, sox_effect_options(e, 1, args);
	sox_add_effect(chain, e, &interm_signal, &out->signal);
	free(e);

	sox_flow_effects(chain, NULL, NULL);

	sox_delete_effects_chain(chain);
	sox_close(out);
	sox_close(in);
}

int main(int argc, char** argv) {
    int i;
    int count = 0;
    static const float a_hp[2] = { -1.99599, 0.99600 };
    static const float b_hp[2] = { -2, 1 };
    float a_sig[2] = { 0 };
    float b_sig[2] = { 0 };
    float mem_hp_x[2] = { 0 };
    float mem_resp_x[2] = { 0 };
    float mem_preemph = 0;
    float x[FRAME_SIZE];
    int gain_change_count = 0;
    FILE* f1;
    FILE* ffeat;
    FILE* fpcm = NULL;
    short pcm[FRAME_SIZE] = { 0 };
    short pcmbuf[FRAME_SIZE * 4] = { 0 };
    int noisebuf[FRAME_SIZE * 4] = { 0 };
    short tmp[FRAME_SIZE] = { 0 };
    float savedX[FRAME_SIZE] = { 0 };
    float speech_gain = 1;
    int last_silent = 1;
    float old_speech_gain = 1;
    int one_pass_completed = 0;
    LPCNetEncState* st;
    float noise_std = 0;
    int training = -1;
    int encode = 0;
    int decode = 0;
    int quantize = 0;

	// ./dump_data -test test_input.s16 test_features.f32
	// ./dump_data -train input.s16 features.f32 data.u8
	// ./dump_data -mode train -i input.s16 -o data.f32
	// ./dump_data -mode test -i *.s16
	// ./dump_data -m test -i input.s16 -o
	cxxopts::Options options("dump_data", "LPCNet program");
	options.add_options()
		("i,input", "input data or path is PCM without header", cxxopts::value<std::string>())
		("o,out", "output path", cxxopts::value<std::string>())
		("m,mode", "train or test or qtrain or qtest", cxxopts::value<std::string>())
		("t,type", "The processing method is designated as <empty> or 't2'", cxxopts::value<std::string>()->default_value("none"))
		;

	std::string input, output, mode, type = "none";

	try
	{
		auto result = options.parse(argc, argv);
		mode = result["m"].as<std::string>();
		if (mode == "train")
			training = 1;
		else if (mode == "test")
			training = 0;
		else if (mode == "qtrain")
		{
			training = 1;
			quantize = 1;
		}
		else if (mode == "qtest")
		{
			training = 0;
			quantize = 1;
		}
		else if (mode == "encode")
		{
			training = 0;
			quantize = 1;
			encode = 1;
		}
		else if (mode == "decode")
		{
			training = 0;
			decode = 1;
		}

		if (result.count("i") == 0)
			throw std::exception("no input arg");
		
		if (result.count("o") == 0)
			throw std::exception("no ouput arg");

		input = result["i"].as<std::string>();
		output = result["o"].as<std::string>();
		type = result["t"].as<std::string>();

		if (result.count("help"))
			throw std::exception("help");
	}
	catch (const std::exception& ex)
	{
		std::cout << options.help() << std::endl;
		std::cout << ex.what() << std::endl;
		std::cout << "usage: ./dump_data --mode train -i ./*.wav or s16 -o ./train" << std::endl;
		std::cout << "usage: ./dump_data --mode test -i ./train/*.s16 -o ./feats" << std::endl;
		exit(0);
	}

	fprintf(stdout, "Mode: %s, Type: %s\n", mode.c_str(), type.c_str());

    st = lpcnet_encoder_create();
	sox_init();

	fs::path input_path(input);
	fs::path output_path(output);
	cppglob::glob_iterator it = cppglob::iglob(input_path), end;
	std::list<fs::path> input_files(it, end);
	fs::create_directories(output_path);
	
	if (training)
	{
		// create training merged data
		if (input_files.size() > 1)
		{
			auto parent = input_path.parent_path();
			if (parent.string() == "" || parent.string() == ".")
				parent = fs::current_path();

			auto parent_path =  parent.string();
			auto parent_name = parent.filename().string();

			fs::path merge = output_path;
			merge.append(parent_name + ".s16.merge");

			f1 = fopen(merge.string().c_str(), "wb");
			if (f1) {
				for (auto& file : input_files)
				{
					fs::path out = output_path;
					out.append(file.filename().string());

					if (file.extension() == ".wav")
					{
						out.replace_extension(".s16");
						convert_to(file, out, "sw");			// remove header and resampling
					}

					fprintf(stdout, "Merge: %s\n", out.string().c_str());
					FILE* to = fopen(out.string().c_str(), "rb");
					assert(to);
					if (to) {
						copy(to, f1);
						fclose(to);
					}
				}
				fclose(f1);
				input_path = merge;
			}
		}
		else
		{
			if (input_path.extension() == ".wav")
			{
				fs::path out = output_path;
				out.append(input_path.filename().string());
				out.replace_extension(".s16");
				convert_to(input_path, out);		// remove header and resampling
				input_path = out;
			}
		}

		input_files.clear();
		input_files.emplace_back(input_path);	// merged 
	}

	for (auto& input_file : input_files)
	{
		lpcnet_encoder_init(st);
		if (type == "t2")
			st->type = 1;

		f1 = fopen(input_file.string().c_str(), "rb");
		if (f1 == NULL) {
			fprintf(stderr, "Error opening input .s16 16kHz speech input file: %s\n", argv[2]);
			exit(1);
		}
		
		fs::path ffeat_path = output_path;
		ffeat_path.append(input_file.filename().string());
		ffeat_path.replace_extension(".f32");

		ffeat = fopen(ffeat_path.string().c_str(), "wb");
		if (ffeat == NULL) {
			fprintf(stderr, "Error opening output feature file: %s\n", argv[3]);
			exit(1);
		}

		if (decode) {
			float vq_mem[NB_BANDS] = { 0 };
			while (1) {
				int ret;
				unsigned char buf[8];
				float features[4][NB_TOTAL_FEATURES];
				//int c0_id, main_pitch, modulation, corr_id, vq_end[3], vq_mid, interp_id;
				//ret = fscanf(f1, "%d %d %d %d %d %d %d %d %d\n", &c0_id, &main_pitch, &modulation, &corr_id, &vq_end[0], &vq_end[1], &vq_end[2], &vq_mid, &interp_id);
				ret = fread(buf, 1, 8, f1);
				if (ret != 8) break;
				decode_packet(features, vq_mem, buf);
				for (i = 0; i < 4; i++) {
					fwrite(features[i], sizeof(float), NB_TOTAL_FEATURES, ffeat);
				}
			}
			return 0;
		}
		if (training) {
			fs::path pcm_path = output_path;
			pcm_path.append(input_file.filename().string());
			pcm_path.replace_extension(".u8");
			fpcm = fopen(pcm_path.string().c_str(), "wb");
			if (fpcm == NULL) {
				fprintf(stderr, "Error opening output PCM file: %s\n", argv[4]);
				exit(1);
			}
		}
		while (1) {
			float E = 0;
			int silent;
			for (i = 0; i < FRAME_SIZE; i++) x[i] = tmp[i];
			fread(tmp, sizeof(short), FRAME_SIZE, f1);
			if (feof(f1)) {
				if (!training) break;
				rewind(f1);
				fread(tmp, sizeof(short), FRAME_SIZE, f1);
				one_pass_completed = 1;
			}
			for (i = 0; i < FRAME_SIZE; i++) E += tmp[i] * (float)tmp[i];
			if (training) {
				silent = E < 5000 || (last_silent && E < 20000);
				if (!last_silent && silent) {
					for (i = 0; i < FRAME_SIZE; i++) savedX[i] = x[i];
				}
				if (last_silent && !silent) {
					for (i = 0; i < FRAME_SIZE; i++) {
						float f = (float)i / FRAME_SIZE;
						tmp[i] = (int)floor(.5 + f * tmp[i] + (1 - f) * savedX[i]);
					}
				}
				if (last_silent) {
					last_silent = silent;
					continue;
				}
				last_silent = silent;
			}
			if (count * FRAME_SIZE_5MS >= 10000000 && one_pass_completed) break;
			if (training && ++gain_change_count > 2821) {
				float tmp;
				speech_gain = pow(10., (-20 + (rand() % 40)) / 20.);
				if (rand() % 20 == 0) speech_gain *= .01;
				if (rand() % 100 == 0) speech_gain = 0;
				gain_change_count = 0;
				rand_resp(a_sig, b_sig);
				tmp = (float)rand() / RAND_MAX;
				noise_std = 10 * tmp * tmp;
			}
			biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
			biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
			preemphasis(x, &mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
			for (i = 0; i < FRAME_SIZE; i++) {
				float g;
				float f = (float)i / FRAME_SIZE;
				g = f * speech_gain + (1 - f) * old_speech_gain;
				x[i] *= g;
			}
			for (i = 0; i < FRAME_SIZE; i++) x[i] += rand() / (float)RAND_MAX - .5;
			/* PCM is delayed by 1/2 frame to make the features centered on the frames. */
			for (i = 0; i < FRAME_SIZE - TRAINING_OFFSET; i++) pcm[i + TRAINING_OFFSET] = float2short(x[i]);
			compute_frame_features(st, x);

			RNN_COPY(&pcmbuf[st->pcount * FRAME_SIZE], pcm, FRAME_SIZE);
			if (fpcm) {
				compute_noise(&noisebuf[st->pcount * FRAME_SIZE], noise_std);
			}
			st->pcount++;
			/* Running on groups of 4 frames. */
			if (st->pcount == 4) {
				unsigned char buf[8];
				process_superframe(st, buf, ffeat, encode, quantize);
				if (fpcm) write_audio(st, pcmbuf, noisebuf, fpcm);
				st->pcount = 0;
			}
			//if (fpcm) fwrite(pcm, sizeof(short), FRAME_SIZE, fpcm);
			for (i = 0; i < TRAINING_OFFSET; i++) pcm[i] = float2short(x[i + FRAME_SIZE - TRAINING_OFFSET]);
			old_speech_gain = speech_gain;
			count++;
		}
		fclose(f1);
		fclose(ffeat);
		if (fpcm) fclose(fpcm);
	}
    lpcnet_encoder_destroy(st);
	sox_quit();
    return 0;
}