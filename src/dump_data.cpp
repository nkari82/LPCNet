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
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <list>
#include <regex>
#include <type_traits>

#ifdef __has_include
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
static_assert(false, "must include filesystem!");
#endif
#endif

#define CPPGLOB_STATIC
#define SAMPLE_RATE 16000

#include <cppglob/glob.hpp>
#include <cppglob/iglob.hpp>
#include <cxxopts.hpp>

extern "C"
{
#include "kiss_fft.h"
#include "common.h"
#include "freq.h"
#include "pitch.h"
#include "arch.h"
#include "celt_lpc.h"
#include "lpcnet.h"
#include "lpcnet_private.h"
}

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

static float get_normalize_gain(const std::list<fs::path>& input_files)
{
	sox_encodinginfo_t out_encoding =
	{
		SOX_ENCODING_UNKNOWN,
		0,
		std::numeric_limits<double>::infinity(),
		sox_option_no,
		sox_option_no,
		sox_option_no,
		sox_false
	};

	sox_signalinfo_t in_signal = { SAMPLE_RATE, 1, 16, 0, NULL };
	sox_signalinfo_t interm_signal;
	double rms_pk_lev_dB = 0.0;

	char* args[10];
	for (auto& path : input_files)
	{
		auto in_ext = path.filename().extension();
		sox_format_t* in = sox_open_read(path.string().c_str(), (in_ext == ".wav") ? NULL : &in_signal, NULL, NULL);
		sox_format_t* out = sox_open_write("", &in->signal, &in->encoding, "null", NULL, NULL);
		interm_signal = in->signal; /* NB: deep copy */

		sox_effects_chain_t* chain = sox_create_effects_chain(&in->encoding, &out_encoding);
		sox_effect_t* e = sox_create_effect(sox_find_effect("input"));
		args[0] = (char *)in, sox_effect_options(e, 1, args);
		sox_add_effect(chain, e, &interm_signal, &out->signal);
		free(e);

		e = sox_create_effect(sox_find_effect("stats"));
		sox_effect_options(e, 0, NULL);
		sox_add_effect(chain, e, &interm_signal, &out->signal);
		free(e);

		e = sox_create_effect(sox_find_effect("output"));
		args[0] = (char *)out, sox_effect_options(e, 1, args);
		sox_add_effect(chain, e, &interm_signal, &out->signal);
		free(e);

		sox_flow_effects(chain, NULL, NULL);

		char buf[512]{ 0 };
		std::setvbuf(stderr, buf, _IOLBF, sizeof(buf));
		sox_delete_effects_chain(chain);

		std::stringstream ss(buf);
		std::string to;
		float pk_lev_dB(0.f);
		if (buf != nullptr)
		{
			while (std::getline(ss, to, '\n'))
			{
				if (to.find("Pk lev dB") == 0)
				{
					to.replace(0, strlen("Pk lev dB"), "");
					pk_lev_dB = std::stof(to);
					break;
				}
			}
		}

		rms_pk_lev_dB += (pk_lev_dB * pk_lev_dB);

		std::memset(buf, 0, 512);
		std::setvbuf(stderr, buf, _IOLBF, sizeof(buf));
		std::setvbuf(stderr, nullptr, _IONBF, 0);

		sox_close(out);
		sox_close(in);
	}

	rms_pk_lev_dB = std::sqrt(rms_pk_lev_dB / (double)input_files.size());
	fprintf(stdout, "RMS Pk lev dB: %f\n", rms_pk_lev_dB);
	return (float)rms_pk_lev_dB;
}

static void convert_to(const fs::path& in_path, const fs::path& out_path, const char* type, const char* trim, const char* pad, float gain)
{
	// find effect
	static const sox_effect_handler_t* input_efh = sox_find_effect("input");
	static const sox_effect_handler_t* output_efh = sox_find_effect("output");
	static const sox_effect_handler_t* gain_efh = sox_find_effect("gain");
	static const sox_effect_handler_t* silence_efh = sox_find_effect("silence");
	static const sox_effect_handler_t* reverse_efh = sox_find_effect("reverse");
	static const sox_effect_handler_t* rate_efh = sox_find_effect("rate");
	static const sox_effect_handler_t* channels_efh = sox_find_effect("channels");
	static const sox_effect_handler_t* pad_efh = sox_find_effect("pad");
	static const auto safe_sox_close = [](sox_format_t* f) { if (f != nullptr) sox_close(f); };

	auto in_ext = in_path.filename().extension();
	sox_encodinginfo_t out_encoding =
	{
		SOX_ENCODING_SIGN2,
		16,
		std::numeric_limits<double>::infinity(),
		sox_option_default,
		sox_option_default,
		sox_option_default,
		sox_false
	};

	sox_signalinfo_t out_signal = { SAMPLE_RATE, 1, 16, 0, NULL };
	sox_signalinfo_t in_signal = { SAMPLE_RATE, 1, 16, 0, NULL };
	sox_signalinfo_t interm_signal;

	std::shared_ptr<sox_format_t> in = { sox_open_read(in_path.string().c_str(), (in_ext == ".wav") ? NULL : &in_signal, NULL, NULL), safe_sox_close };
	std::shared_ptr<sox_format_t> out = { sox_open_write(out_path.string().c_str(), &out_signal, &out_encoding, type, NULL, NULL), safe_sox_close };
	if (out.get() == nullptr) { std::cerr << "failed open out" << std::endl; return; }
	interm_signal = in->signal; /* NB: deep copy */

	std::shared_ptr<sox_effects_chain_t> chain = { sox_create_effects_chain(&in->encoding, &out->encoding), sox_delete_effects_chain };
	std::shared_ptr<sox_effect_t> e;
	{
		e = { sox_create_effect(input_efh), free };
		const char* args[]{ (const char*)in.get() };
		sox_effect_options(e.get(), 1, (char**)args);
		sox_add_effect(chain.get(), e.get(), &interm_signal, &in->signal);
	}

	if (in->signal.rate != out->signal.rate)
	{
		e = { sox_create_effect(rate_efh), free };
		sox_effect_options(e.get(), 0, NULL);
		sox_add_effect(chain.get(), e.get(), &interm_signal, &out->signal);
	}

	if (in->signal.channels != out->signal.channels)
	{
		e = { sox_create_effect(channels_efh), free };
		sox_effect_options(e.get(), 0, NULL);
		sox_add_effect(chain.get(), e.get(), &interm_signal, &out->signal);
	}

	if (gain != 0.0f)
	{
		std::string option = std::to_string(gain);
		e = { sox_create_effect(gain_efh), free };
		const char* args[]{ option.c_str() };
		sox_effect_options(e.get(), 1, (char**)args);
		sox_add_effect(chain.get(), e.get(), &interm_signal, &out->signal);
	}

	// https://digitalcardboard.com/blog/2009/08/25/the-sox-of-silence/comment-page-2/
	// Trimming all (silence 1 0.1 1% -1 0.1 1%)
	// Trimming begin (silence 1 0.1 1%)
	// [-l] above-periods [duration threshold[d|%] [below-periods duration threshold[d|%]]
	if (trim != nullptr && std::strlen(trim) > 0)
	{
		std::stringstream ss(trim);
		std::vector<std::string> tokens{ "2", "1%", "1%" };
		for (int i = 0; i < 3 && std::getline(ss, tokens[i], ':');)
			if (!tokens[i].empty()) i++;

		int silence = std::stoi(tokens[0]);
		switch (silence)
		{
		case 1: // begin
		{
			e = { sox_create_effect(silence_efh), free };
			const char* args[]{ "1", "0.1", tokens[1].c_str() };
			sox_effect_options(e.get(), 3, (char**)args);
			sox_add_effect(chain.get(), e.get(), &interm_signal, &out->signal);
			break;
		}
		case 2: // begin + end
		{
			for (int i = 1; i < 3; ++i)
			{
				e = { sox_create_effect(silence_efh), free };
				const char* args[]{ "1", "0.1", tokens[i].c_str() };
				sox_effect_options(e.get(), 3, (char**)args);
				sox_add_effect(chain.get(), e.get(), &interm_signal, &out->signal);

				e = { sox_create_effect(reverse_efh), free };
				sox_effect_options(e.get(), 0, NULL);
				sox_add_effect(chain.get(), e.get(), &interm_signal, &out->signal);
			}
			break;
		}
		case 3: // all
		{
			e = { sox_create_effect(silence_efh), free };
			const char* args[]{ "1", "0.1", tokens[1].c_str(), "-1", "0.1", tokens[1].c_str() };
			sox_effect_options(e.get(), 6, (char**)args);
			sox_add_effect(chain.get(), e.get(), &interm_signal, &out->signal);
			break;
		}
		default:
			break;
		}
	}

	// pad
	if (pad != nullptr && std::strlen(pad) > 0)
	{
		std::stringstream ss(pad);
		std::vector<std::string> tokens{ "0", "0" };
		for (int i = 0; i < 2 && std::getline(ss, tokens[i], ':');)
			if (!tokens[i].empty()) i++;

		e = { sox_create_effect(pad_efh), free };
		const char* args[]{ tokens[0].c_str(), tokens[1].c_str() };
		sox_effect_options(e.get(), 2, (char**)args);
		sox_add_effect(chain.get(), e.get(), &interm_signal, &out->signal);
	}

	{
		e = { sox_create_effect(output_efh), free };
		const char* args[]{ (const char*)out.get() };
		sox_effect_options(e.get(), 1, (char**)args);
		sox_add_effect(chain.get(), e.get(), &interm_signal, &out->signal);
	}

	sox_flow_effects(chain.get(), NULL, NULL);
}

void show_help(const cxxopts::Options& options, const char* message = nullptr)
{
	std::cout << options.help() << std::endl;
	std::cout << "usage: ./dump_data -m train -i \"./*.wav or s16\" -o ./train" << std::endl;
	std::cout << "usage: ./dump_data -m test -f <0 or 1> -i \"./train/*.wav or *.s16\" -o ./dump" << std::endl;
	
	if(message != nullptr)
		std::cout << message << std::endl;
}

/*
LPCNet acoustic feature
features[:18] : 18-dim Bark scale cepstrum
features[18:36] : Not used
features[36:37] : pitch period
features[37:38] : pitch correlation
features[39:55] : LPC
window_size (=n_fft) = 320 (20ms)
frame_shift(=hop_size) = 160 (10ms)

examples
-m test -f 1 -t 2:0.5%:0.5% -p 0:0.01 -i ./tsuchiya/tsuchiya_angry/*.wav -o ./tsuchiya/tsuchiya_angry/dump
-m test -f 1 -t 2:0.5%:0.5% -p 0:0.01 -i D:/Voice/basic5000/wav/*.wav -o D:/Voice/basic5000/dump
-m train -i D:/Github/LPCNet/bin/datasets/jsut/basic/pcm/*.s16 -o D:/Voice/basic5000/train
*/

int main(int argc, const char** argv) {
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
	FILE* fm;
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
	int format = 0;
	int normalize = 0;
	int pcount = 0;
	std::string input;
	std::string output;
	std::string mode;
	std::string trim;
	std::string pad;

	cxxopts::Options options("dump_data", "LPCNet program");
	options.add_options()
		("i,input", "input data or path is PCM without header", cxxopts::value<std::string>())
		("o,out", "output path", cxxopts::value<std::string>())
		("m,mode", "train or test or qtrain or qtest", cxxopts::value<std::string>())
		("f,format", "If '1', the output format is 'bark bands[18] + pitch period[19] and correlation[20]'", cxxopts::value<int>()->default_value("0"))
		("t,trim", "'WAV' file silent section trimming, '1' is begin, 2 is begin and end, '2' is all and threshold[:d%]", cxxopts::value<std::string>()->default_value("1:1%")) // best 2:0.5%:0.5%
		("p,pad", "'WAV' file padding '0:1.5' is add 0 and 1.5 seconds to the begin and end", cxxopts::value<std::string>()->default_value("0:0")) // best 0:0.01 (10ms)
		("n,norm", "normalize '1' is enable", cxxopts::value<int>()->default_value("0"))
		;

	if (argc < 2)
	{
		show_help(options);
		exit(0);
	}

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

		trim = result["t"].as<std::string>();
		pad = result["p"].as<std::string>();

		if (result.count("i") == 0)
		{
			show_help(options, "no input arg");
			exit(0);
		}

		if (result.count("o") == 0)
		{
			show_help(options, "no ouput arg");
			exit(0);
		}

		input = result["i"].as<std::string>();
		output = result["o"].as<std::string>();
		format = result["f"].as<int>();
		normalize = result["n"].as<int>();

		if (result.count("help"))
		{
			show_help(options);
			exit(0);
		}
	}
	catch (const std::exception& ex)
	{
		show_help(options, ex.what());
		exit(0);
	}

	fprintf(stdout, "Mode: %s \n", mode.c_str());
	switch (format)
	{
	case 1:
		fprintf(stdout, "the output format is 'bark bands[18] + pitch period[19] and correlation[20]'\n");
		break;
	default:
		break;
	}

	st = lpcnet_encoder_create();
	sox_init();

	fs::path input_path(input);
	fs::path output_path(output);
	fs::path output_path_feats(output);
	fs::path output_path_pcm(output);

	cppglob::glob_iterator it = cppglob::iglob(input_path), end;
	std::list<fs::path> input_files(it, end);
	fs::create_directories(output_path);

	if (!training)
	{
		output_path_feats.append("feats");
		fs::create_directories(output_path_feats);

		output_path_pcm.append("pcm");
		if (!input_files.empty() && input_files.front().extension() == ".wav")
			fs::create_directories(output_path_pcm);
	}
	
	float gain = normalize ? get_normalize_gain(input_files) : 0.0f;

	if (training)
	{
		// create training merged data
		auto parent = input_path.parent_path();
		if (parent.string() == "" || parent.string() == ".")
			parent = fs::current_path();

		auto parent_path = parent.string();
		auto parent_name = parent.filename().string();

		fs::path merge = output_path;
		merge.append(parent_name + ".s16.merge");

		fm = fopen(merge.string().c_str(), "wb");
		if (fm) {
			for (auto& file : input_files)
			{
				FILE* to(nullptr);
				auto in_ext = file.filename().extension();
				if (in_ext == ".wav")
				{
					fs::path out = output_path;
					out.append(file.filename().string());
					out.replace_extension(".s16");

					fprintf(stdout, "Convert: %s\r", file.string().c_str());
					fflush(stdout);
					convert_to(file, out, "sw", trim.c_str(), pad.c_str(), gain);
					to = fopen(out.string().c_str(), "rb");
				}
				else
				{
					to = fopen(file.string().c_str(), "rb");
				}
				assert(to);
				if (to) {
					copy(to, fm);
					fclose(to);
				}
			}
			fclose(fm);
			input_path = merge;
		}

		input_files.clear();
		input_files.emplace_back(input_path);	// merged 
	}

	for (auto& input_file : input_files)
	{
		lpcnet_encoder_init(st);
		count = 0;
		pcount = 0;

		fprintf(stdout, "Process file: %ws\r", input_file.c_str());
		fflush(stdout);

		auto in_ext = input_file.extension();
		if (in_ext == ".wav")
		{
			fs::path pcm_path = output_path_pcm;
			pcm_path.append(input_file.filename().string());
			pcm_path.replace_extension(".s16");

			fprintf(stdout, "\nConvert: %s\r\b\r", input_file.string().c_str());
			fflush(stdout);
			convert_to(input_file, pcm_path, "sw", trim.c_str(), pad.c_str(), gain);
			input_file = pcm_path;
		}

		f1 = fopen(input_file.string().c_str(), "rb");
		if (f1 == NULL) {
			fprintf(stderr, "Error opening input .s16 16kHz speech input file: %s\n", input_file.string().c_str());
			exit(1);
		}
		
		fs::path ffeat_path = output_path_feats;
		ffeat_path.append(input_file.filename().string());
		ffeat_path.replace_extension(".f32");

		ffeat = fopen(ffeat_path.string().c_str(), "wb");
		if (ffeat == NULL) {
			fprintf(stderr, "Error opening output feature file: %s\n", ffeat_path.string().c_str());
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
				fprintf(stderr, "Error opening output PCM file: %s\n", pcm_path.string().c_str());
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
			
			if (fpcm) {
				RNN_COPY(&pcmbuf[st->pcount * FRAME_SIZE], pcm, FRAME_SIZE);
				compute_noise(&noisebuf[st->pcount * FRAME_SIZE], noise_std);
			}
			
			/* Running on groups of 4 frames. */
			if (++st->pcount == 4) {
				unsigned char buf[8];
				process_superframe(st, buf, ffeat, encode, quantize, format);
				if (fpcm) write_audio(st, pcmbuf, noisebuf, fpcm);
				pcount += st->pcount;
				st->pcount = 0;
			}

			//if (fpcm) fwrite(pcm, sizeof(short), FRAME_SIZE, fpcm);
			for (i = 0; i < TRAINING_OFFSET; i++) pcm[i] = float2short(x[i + FRAME_SIZE - TRAINING_OFFSET]);
			old_speech_gain = speech_gain;
			count++;
		}
		if (fpcm) fclose(fpcm);
		fclose(f1);
		fclose(ffeat);
	}
	lpcnet_encoder_destroy(st);
	sox_quit();
	return 0;
}