# embla-metal

A Metal engine for [Embla 1](https://huggingface.co/sevren-ai/Embla1-4bit) on Apple silicon, tuned
for the M5 Pro and the iPhone 18 Pro. No dependencies: the 4-bit weights are memory-mapped
straight into Metal, the engram table is read from the SSD, and each token is one pipelined pass
of a few wide kernels.

## Chat with it on a Mac

On a Mac with Apple silicon, macOS 15 or later and Xcode (or the Command Line Tools,
`xcode-select --install`):

```bash
git clone https://github.com/sevren-ai/embla-metal && cd embla-metal
swift run -c release embla download        # the model, 5.6 GB, into the Hugging Face cache
swift run -c release embla serve --open    # a local server, and the chat page in your browser
```

The first `swift run` builds for a minute or so. `serve` listens only on this Mac at
`http://127.0.0.1:8080`: a chat page at `/` and an OpenAI-compatible API at `/v1`
(`chat/completions`, `completions`, `models`; streamed or not). OpenAI clients can use
`base_url="http://127.0.0.1:8080/v1"`; an API key is not required. Options: `--port`, `--host`,
`--context` (the longest conversation in tokens, default 32768) and `--model` (another repo in
the cache, or a directory).

**Keep the server on its default loopback address.** It has no authentication or TLS and permits
cross-origin browser requests, so an API key provides no protection. Do not expose it directly to
a local network or the internet. If remote access is necessary, put it behind an appropriately
configured authenticated TLS reverse proxy and restrict access to trusted clients.

Embla 1 thinks before it answers: the reasoning streams as `reasoning_content`, the answer as
`content`. Sampling defaults to temperature 0.6 and top-p 0.95; `temperature: 0` is greedy. One
request runs at a time.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the capital of Denmark?"}]}'
```

## Speed

Tokens a second by context, on real text, 128 new tokens, November 2026:

| context | 1k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|
| M5 Pro, decode | 339 | 328 | 321 | 311 | 290 |
| M5 Pro, prompt | 10.4k | 12.2k | 12.6k | 12.6k | 12.1k |
| iPhone 18 Pro, decode | 166 | 163 | 161 | 155 | 145 |

A 1024-token prompt then 256 new tokens: 862 ms on the M5 Pro, 1772 ms on the iPhone 18 Pro.
Prompts on the iPhone run at 4.7-5.1k tokens a second. Sampling costs ~2% against greedy decoding.

## How it works

- **Decode:** one pass a token of ~220 dependent dispatches, the next token's work encoded while
  the current one runs; the CPU reads the next token's engram rows from the SSD in between.
- **Prefill:** batched, up to 2048 tokens at a time, the matrices as 4-bit GEMMs on the neural
  accelerators (Metal 4 tensor ops) where the GPU has them (M5 and A19 on), simdgroup matrices
  before. `EMBLA_NAX=0` or `1` overrides the detection.
- **Attention:** sliding, full and sparse layers; the full layers' keys and values cached in int8;
  past 1152 tokens the sparse layers attend the 1024 highest-scoring keys and the 128 latest.
- **Sampling:** temperature and top-p in the step's last kernel (Gumbel-max over the nucleus), so
  decoding stays pipelined.
- **Checked** against a reference implementation's logits step by step (`embla-bench check`).

## On an iPhone

There is no iPhone app here: the engine builds for iOS 18 or later, and an app of your own links
`EmblaMetal` (or `EmblaBenchKit`, whose `runBench` takes an `embla-bench` command's arguments).
`EMBLA_*` settings come as the app's environment variables; `EMBLA_KERNELS=<dir>` (relative to
the app's Documents) loads `<name>.metal` files from there instead of the built-in ones.

## Development

```bash
swift build -c release
swift test
M=$(ls -d ~/.cache/huggingface/hub/models--sevren-ai--Embla1-4bit/snapshots/*/)
.build/release/embla-bench e2e $M 1024 256 5     # a 1024-token prompt + 256 new tokens
.build/release/embla-bench speed $M 256          # decode tok/s
.build/release/embla-bench det $M 64             # the same tokens twice: any logit that differs
.build/release/embla-bench check $M tokens.txt out/   # every step's logits
```

Swift 6 (Xcode 16 or later). The libraries: `EmblaMetal` (the engine), `EmblaChat` (tokenizer,
chat template, download, server) and `EmblaBenchKit` (the bench commands).

## License

MIT (see [LICENSE](LICENSE)). `Sources/EmblaMetal/Kernels/nax.metal` incorporates MLX 0.31.0's
steel NAX tiles and row operations, Copyright Apple Inc., also under MIT. The full Sevren and
Apple copyright and license notices are in `LICENSE`. The model weights have their own license on
Hugging Face and are not covered by this repository's MIT license.
