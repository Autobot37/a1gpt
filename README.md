## modifying a1gpt for GPT-XL
`$ python3 scripts/download_and_convert_gpt2.py`

```
$ mkdir build
$ cd build
$ cmake ..

$ make -j
$ ./gpt2 -h
```

```
$ ./gpt2 -s 1688452945 -n 256
a1gpt seed=1688452945 sampling_temperature=0.90 ntokens=301
```
