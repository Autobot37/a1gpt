import mmap
import json
import os
import numpy as np
from huggingface_hub import hf_hub_download

# are we running from the scripts directory? if so, move up a level
if os.path.basename(os.getcwd()) == "scripts":
    os.chdir("..")


def convert_safetensors(filename, bin_filename, cpp_filename):
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)
            header_offset = n + 8
            bin_file = open(bin_filename, "wb")
            cpp_file = open(cpp_filename, "w")
            bin_offset = 0

            def add_weight_data(c_name, shape, buf):
                shape_str = ", ".join([str(x) for x in shape])
                cpp_file.write(f"  {c_name:30} = Tensorf<{len(shape)}>((float*)(data + 0x{bin_offset:08x}), {shape_str});\n")
                bin_file.write(buf)
                siz = len(buf)
                padded_siz = (siz+31) & (~31)  # pad all weights to multiples of 32 bytes
                if padded_siz > siz:
                    print("padding {} from {} to {}".format(c_name, siz, padded_siz))
                    bin_file.write(bytearray(padded_siz - siz))
                return padded_siz

            def add_weights(c_name, meta_name):
                offs = metadata[meta_name]["data_offsets"]
                shape = metadata[meta_name]["shape"]
                off = offs[0] + header_offset
                siz = offs[1] - offs[0]
                return add_weight_data(c_name, shape, m[off:off+siz])

            def add_weights_transposed(c_name, meta_name):
                offs = metadata[meta_name]["data_offsets"]
                off = offs[0] + header_offset
                siz = offs[1] - offs[0]
                w = np.frombuffer(m[off:off+siz], dtype=np.float32)
                w = w.reshape(metadata[meta_name]["shape"])
                w = w.T.copy()
                return add_weight_data(c_name, list(w.shape), w.tobytes())

            embed_dim = metadata["wte.weight"]["shape"][1]
            context_len = metadata["wpe.weight"]["shape"][0]
            ntokens = metadata["wte.weight"]["shape"][0]
            nlayers = 48
            print(embed_dim, context_len, ntokens, nlayers)

            cpp_file.write(f'''// This file was generated by scripts/model_load.py
// DO NOT EDIT

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "model.h"
#include "tensor.h"

bool load_gpt2_model(Model &m) {{
  const char *fname = "model/gpt2-weights.bin";
  int fd = open(fname, O_RDONLY);
  if (fd < 0) {{
    perror(fname);
    return false;
  }}
  struct stat sb;
  fstat(fd, &sb);
  char *data = (char*) mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

  m.mmap_data = data;
  m.mmap_siz = sb.st_size;

  m.embedding_dim = {embed_dim};
  m.context_len = {context_len};
  m.ntokens = {ntokens};
  m.h = new TransformerBlock[{nlayers}];
''')
            bin_offset += add_weights("m.wte_weight", "wte.weight")
            bin_offset += add_weights("m.wpe_weight", "wpe.weight")
            bin_offset += add_weights("m.ln_f.bias", "ln_f.bias")
            bin_offset += add_weights("m.ln_f.weight", "ln_f.weight")
            for i in range(48):
                bin_offset += add_weights(f"m.h[{i}].ln_1.bias", f"h.{i}.ln_1.bias")
                bin_offset += add_weights(f"m.h[{i}].ln_1.weight", f"h.{i}.ln_1.weight")
                cpp_file.write(f"  m.h[{i}].attn.num_heads = 25;\n")
                bin_offset += add_weights(f"m.h[{i}].attn.c_attn_bias", f"h.{i}.attn.c_attn.bias")
                bin_offset += add_weights_transposed(f"m.h[{i}].attn.c_attn_weight", f"h.{i}.attn.c_attn.weight")
                bin_offset += add_weights(f"m.h[{i}].attn.c_proj_bias", f"h.{i}.attn.c_proj.bias")
                bin_offset += add_weights_transposed(f"m.h[{i}].attn.c_proj_weight", f"h.{i}.attn.c_proj.weight")
                bin_offset += add_weights(f"m.h[{i}].ln_2.bias", f"h.{i}.ln_2.bias")
                bin_offset += add_weights(f"m.h[{i}].ln_2.weight", f"h.{i}.ln_2.weight")
                bin_offset += add_weights(f"m.h[{i}].mlp.c_fc_bias", f"h.{i}.mlp.c_fc.bias")
                bin_offset += add_weights_transposed(f"m.h[{i}].mlp.c_fc_weight", f"h.{i}.mlp.c_fc.weight")
                bin_offset += add_weights(f"m.h[{i}].mlp.c_proj_bias", f"h.{i}.mlp.c_proj.bias")
                bin_offset += add_weights_transposed(f"m.h[{i}].mlp.c_proj_weight", f"h.{i}.mlp.c_proj.weight")
            cpp_file.write('''
  close(fd);
  return true;
}
''')


def main():
    filename = hf_hub_download("gpt2-xl", filename="model.safetensors")
    convert_safetensors(filename, "model/gpt2-weights.bin", "model_load_gpt2.cpp")


if __name__ == "__main__":
    main()
