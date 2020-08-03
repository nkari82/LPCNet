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

#include <math.h>
#include <stdio.h>

#if defined(__cplusplus)
#include <iostream>
#include <string>
#include <vector>
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

extern "C"
{
#endif
#include "arch.h"
#include "lpcnet.h"
#include "freq.h"
#if defined(__cplusplus)
}
#endif

int main(int argc, char** argv) {
	int mode = 0, in = -1, out = -1;
    FILE* fin, * fout;
    LPCNetState* net;
    net = lpcnet_create();

    if (argc == 4 && strcmp(argv[1], "-taco") == 0)
	{
		mode = 1; 
		in = 2; 
		out = 3; // taco
	}
	
    if (argc == 3)
    {
        in = 1;
        out = 2;
    }
    
    if (in == -1 || out == -1)
    {
        fprintf(stderr, "usage: test_lpcnet <empty or -taco> <features.f32> <output.pcm>\n");
        return 0;
    }

    fin = fopen(argv[in], "rb");
    if (fin == NULL) {
        fprintf(stderr, "Can't open %s\n", argv[in]);
        exit(1);
    }

    fout = fopen(argv[out], "wb");
    if (fout == NULL) {
        fprintf(stderr, "Can't open %s\n", argv[out]);
        exit(1);
    }

	fprintf(stdout, "Mode: %d\n", mode);
	
    while (1) {
		float in_features[NB_TOTAL_FEATURES];
		float features[NB_FEATURES];
		short pcm[FRAME_SIZE];
        memset(in_features, 0, sizeof(in_features));
        memset(features, 0, sizeof(features));

        if (mode == 1)
        {
            fread(in_features, sizeof(features[0]), NB_BANDS + 2, fin);
            if (feof(fin)) break;
            RNN_COPY(features, in_features, NB_BANDS);
            RNN_CLEAR(&features[NB_BANDS], NB_BANDS);
            RNN_COPY(&features[NB_BANDS], in_features + NB_BANDS, 2);
        }
        else
        {
            fread(in_features, sizeof(features[0]), NB_TOTAL_FEATURES, fin);
            if (feof(fin)) break;
            RNN_COPY(features, in_features, NB_FEATURES);
            RNN_CLEAR(&features[NB_BANDS], NB_BANDS);
        }

        lpcnet_synthesize(net, features, pcm, FRAME_SIZE);
        fwrite(pcm, sizeof(pcm[0]), FRAME_SIZE, fout);
    }

    fclose(fin);
    fclose(fout);
    lpcnet_destroy(net);
    return 0;
}