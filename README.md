# llama.swift

A package that provides access to
[llama.cpp](https://github.com/ggml-org/llama.cpp)
in your Swift projects.
It re-exports the llama.cpp C++ APIs directly,
using the precompiled XCFramework provided by the llama.cpp project.

The package automatically follows upstream
[llama.cpp version releases](https://github.com/ggml-org/llama.cpp/releases/latest)
and uses [Swift Package Manager versioning](https://docs.swift.org/swiftpm/documentation/packagemanagerdocs/addingdependencies#Constraining-dependency-versions),
so you can use `.upToNextMajor(from:)` or `.exact(_:)` version requirements.

> [!TIP]
> You can add the XCFramework binary dependency directly to your project
> without this package.
>
> For instructions,
> see the [llama.cpp README](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#xcframework)

## Requirements

- Swift 6.0+
- macOS 13.3+ (Apple silicon or Intel)
- iOS 16.4+ (devices only)

The upstream XCFramework has two slices:
`macos-arm64_x86_64` and `ios-arm64`.
Since [ggml-org/llama.cpp#27252](https://github.com/ggml-org/llama.cpp/pull/27252),
llama.cpp no longer builds slices for the iOS Simulator, tvOS, or visionOS,
and it has never built one for watchOS.
A build for one of those platforms fails with an error such as
"While building for iOS Simulator, no library for this platform was found".
`Package.swift` still lists tvOS, watchOS, and visionOS,
but the package does not build for them.
The minimum versions above are the deployment targets of the upstream binaries.

## Installation

Add this package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/llama.swift", .upToNextMajor(from: "3.0.0"))
]
```

Or add it through Xcode's Package Manager.

## Versioning

Each llama.swift release follows a llama.cpp
[version release](https://github.com/ggml-org/ggml/discussions/1579).
For llama.cpp `vX.Y.Z`, the llama.swift version is `(X+2).Y.Z`.
For example, llama.swift `3.0.0` uses llama.cpp `v1.0.0`.
The major version is offset by two because llama.swift 1.x and 2.x already exist.
A llama.cpp version release has no XCFramework of its own,
so the package uses the XCFramework from the nightly build
that the version release names in its `nightly-tag.txt` asset.
`Package.swift` records the llama.cpp version above the XCFramework URL.

A scheduled workflow runs `scripts/sync-llama-release.sh` to make these updates.
To test that script without network access,
run `scripts/test-sync-llama-release.sh`,
which needs `bash`, `git`, `curl`, `jq`, and `python3`.

Versions 1.x and 2.x followed each llama.cpp nightly build,
with the build number as the minor version.
For example, `2.10549.0` uses build `b10549`.
The 2.x versions end at `2.10549.0`.

## Migrating from 2.x

llama.cpp `v1.0.0` removes functions that earlier builds deprecated.
The table below lists the replacement that each deprecation attribute
in `llama.h` at `b10549` names.

| Removed function                  | Replacement                                 |
| --------------------------------- | ------------------------------------------- |
| `llama_add_bos_token`             | `llama_vocab_get_add_bos`                   |
| `llama_add_eos_token`             | `llama_vocab_get_add_eos`                   |
| `llama_copy_state_data`           | `llama_state_get_data`                      |
| `llama_free_model`                | `llama_model_free`                          |
| `llama_get_state_size`            | `llama_state_get_size`                      |
| `llama_load_model_from_file`      | `llama_model_load_from_file`                |
| `llama_load_session_file`         | `llama_state_load_file`                     |
| `llama_n_ctx_train`               | `llama_model_n_ctx_train`                   |
| `llama_n_embd`                    | `llama_model_n_embd`                        |
| `llama_n_head`                    | `llama_model_n_head`                        |
| `llama_n_layer`                   | `llama_model_n_layer`                       |
| `llama_n_vocab`                   | `llama_vocab_n_tokens`                      |
| `llama_new_context_with_model`    | `llama_init_from_model`                     |
| `llama_sampler_init_grammar_lazy` | `llama_sampler_init_grammar_lazy_patterns`  |
| `llama_save_session_file`         | `llama_state_save_file`                     |
| `llama_set_state_data`            | `llama_state_set_data`                      |
| `llama_set_warmup`                | None (see below)                            |
| `llama_token_bos`                 | `llama_vocab_bos`                           |
| `llama_token_cls`                 | `llama_vocab_bos` (see below)               |
| `llama_token_eos`                 | `llama_vocab_eos`                           |
| `llama_token_eot`                 | `llama_vocab_eot`                           |
| `llama_token_fim_mid`             | `llama_vocab_fim_mid`                       |
| `llama_token_fim_pad`             | `llama_vocab_fim_pad`                       |
| `llama_token_fim_pre`             | `llama_vocab_fim_pre`                       |
| `llama_token_fim_rep`             | `llama_vocab_fim_rep`                       |
| `llama_token_fim_sep`             | `llama_vocab_fim_sep`                       |
| `llama_token_fim_suf`             | `llama_vocab_fim_suf`                       |
| `llama_token_get_attr`            | `llama_vocab_get_attr`                      |
| `llama_token_get_score`           | `llama_vocab_get_score`                     |
| `llama_token_get_text`            | `llama_vocab_get_text`                      |
| `llama_token_is_control`          | `llama_vocab_is_control`                    |
| `llama_token_is_eog`              | `llama_vocab_is_eog`                        |
| `llama_token_nl`                  | `llama_vocab_nl`                            |
| `llama_token_pad`                 | `llama_vocab_pad`                           |
| `llama_token_sep`                 | `llama_vocab_sep`                           |
| `llama_vocab_cls`                 | `llama_vocab_bos`                           |

The header names no replacement function for `llama_set_warmup`.
Its deprecation message says that
"user code should do warmup runs manually".
The header names `llama_vocab_cls` as the replacement for `llama_token_cls`,
but `v1.0.0` also removes `llama_vocab_cls`
and names `llama_vocab_bos` as its replacement.

Two multimodal helper functions in `mtmd-helper.h` take a new last argument:
`mtmd_helper_bitmap_init_from_buf` and `mtmd_helper_bitmap_init_from_file`
take a `mtmd_helper_init_opt`,
which you can get from `mtmd_helper_init_opt_default()`.
In `mtmd.h`, `mtmd_gen_audio_process` takes a `mtmd_gen_state *`
as its second argument;
create one with `mtmd_gen_state_init` and free it with `mtmd_gen_state_free`.

These types also change in ways that can break source code:

- `llama_context_params` no longer has `defrag_thold`,
  which the header at `b10549` already marked as deprecated.
  Remove any assignment to it.
- `mtmd_helper_post_decode_callback` receives
  a `const struct mtmd_helper_embd_batch *`
  instead of a `struct llama_batch`.
  The new struct has the decoded embeddings, their positions,
  and the sequence ID.
- `mtmd_gen_inp` no longer has `seed`, `state_data`, or `state_size`,
  and it has the new fields `penalty_repeat` and `penalty_last_n`.
  `mtmd_gen_out` no longer has `state_data` or `state_size`.
  Set the seed in the `mtmd_gen_state_params`
  that you pass to `mtmd_gen_state_init`.
  Keep that `mtmd_gen_state` for the whole generation
  and pass it to each `mtmd_gen_audio_process` call,
  instead of passing state data from one call to the next.
- Several structs have new fields, including `llama_model_params`,
  `llama_context_params`, `llama_model_quantize_params`,
  `llama_perf_context_data`, `mtmd_caps`, `mtmd_gen_audio_info`,
  `mtmd_helper_gen_audio_inp`, and `mtmd_helper_video_init_params`.
  Swift code that calls the full memberwise initializer of one of these structs
  must add the new arguments.
  Code that starts from a function such as `llama_context_default_params()`
  needs no change.

### Saved state

`LLAMA_SESSION_VERSION` changes from 9 to 11,
and `LLAMA_STATE_SEQ_VERSION` changes from 2 to 4.
In llama.cpp `v1.0.0`, `llama_state_load_file` and `llama_state_seq_load_file`
accept only files with the current version,
so they do not load session or sequence files saved with llama.swift 2.x.
`llama_state_load_file` returns `false`,
and `llama_state_seq_load_file` returns 0.

The buffers from `llama_state_get_data` and `llama_state_seq_get_data`
have no version number,
but both version changes changed the KV cache data in them
([ggml-org/llama.cpp#27762](https://github.com/ggml-org/llama.cpp/pull/27762)
and [ggml-org/llama.cpp#28498](https://github.com/ggml-org/llama.cpp/pull/28498)).
Do not restore buffers saved with llama.swift 2.x.
Process the prompt again to create new state.

## Usage

```swift
import LlamaSwift

// MARK: - Setup

// Initialize the backend
llama_backend_init()
defer { llama_backend_free() }

// Load model
let modelParams = llama_model_default_params()
guard let model = llama_model_load_from_file("/path/to/model.gguf", modelParams) else {
    fatalError("Failed to load model")
}
defer { llama_model_free(model) }

// Create context with configuration
var contextParams = llama_context_default_params()
contextParams.n_ctx = 2048    // Context window size
contextParams.n_batch = 512   // Batch size for processing

guard let context = llama_init_from_model(model, contextParams) else {
    fatalError("Failed to create context")
}
defer { llama_free(context) }

// MARK: - Tokenization

// Get vocabulary for tokenization
let vocab = llama_model_get_vocab(model)

// Prepare input text
let prompt = "The future of artificial intelligence is"
let utf8Count = Int32(prompt.utf8.count)

// Tokenize the input prompt.
// A negative result is the number of tokens needed,
// so resize the buffer and try again.
var tokens = [llama_token](repeating: 0, count: Int(utf8Count) + 2)
var tokenCount: Int32 = 0
for _ in 0..<2 {
    tokenCount = llama_tokenize(
        vocab,
        prompt,
        utf8Count,
        &tokens,
        Int32(tokens.count),
        true,  // add_special: add BOS and EOS if the model is configured to
        true   // parse_special: parse special and control tokens
    )
    guard tokenCount < 0, tokenCount != Int32.min else { break }
    tokens = [llama_token](repeating: 0, count: Int(-tokenCount))
}

guard tokenCount > 0 else {
    fatalError("Failed to tokenize prompt")
}

let promptTokens = Array(tokens.prefix(Int(tokenCount)))
guard promptTokens.count <= Int(contextParams.n_batch) else {
    fatalError("Prompt is longer than the batch size")
}

// MARK: - Initial Evaluation

// Create batch for processing
var batch = llama_batch_init(Int32(contextParams.n_batch), 0, 1)
defer { llama_batch_free(batch) }

// Prepare batch with prompt tokens
batch.n_tokens = Int32(promptTokens.count)

for i in 0..<promptTokens.count {
    let idx = Int(i)
    batch.token[idx] = promptTokens[idx]
    batch.pos[idx] = Int32(i)
    batch.n_seq_id[idx] = 1

    if let seq_ids = batch.seq_id, let seq_id = seq_ids[idx] {
        seq_id[0] = 0
    }

    batch.logits[idx] = 0
}

// Only compute logits for the last token
if batch.n_tokens > 0 {
    batch.logits[Int(batch.n_tokens) - 1] = 1
}

// Evaluate the prompt
guard llama_decode(context, batch) == 0 else {
    fatalError("llama_decode failed")
}

// MARK: - Text Generation

var generatedBytes: [UInt8] = []
var n_cur = batch.n_tokens

// Generate up to 100 tokens
for _ in 0..<100 {
    // Get logits for the last token
    guard let logits = llama_get_logits_ith(context, batch.n_tokens - 1) else {
        fatalError("Failed to get logits")
    }

    // Simple greedy sampling (choose highest probability token)
    let vocabSize = llama_vocab_n_tokens(vocab)
    var maxLogit = logits[0]
    var nextToken: llama_token = 0

    for i in 1..<Int(vocabSize) {
        if logits[i] > maxLogit {
            maxLogit = logits[i]
            nextToken = llama_token(i)
        }
    }

    // Check for end of generation (such as end of sequence or end of turn)
    if llama_vocab_is_eog(vocab, nextToken) {
        break
    }

    // Convert token to text.
    // The piece has no null terminator, and a negative result is the
    // buffer size needed. A piece can end partway through a UTF-8 character,
    // so collect bytes and decode them at the end.
    var buffer = [CChar](repeating: 0, count: 32)
    var length = llama_token_to_piece(vocab, nextToken, &buffer, Int32(buffer.count), 0, false)
    if length < 0 {
        buffer = [CChar](repeating: 0, count: Int(-length))
        length = llama_token_to_piece(vocab, nextToken, &buffer, Int32(buffer.count), 0, false)
    }

    if length > 0 {
        generatedBytes += buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }
    }

    // Prepare batch for the next token
    batch.n_tokens = 1
    batch.token[0] = nextToken
    batch.pos[0] = n_cur
    batch.n_seq_id[0] = 1

    if let seq_ids = batch.seq_id, let seq_id = seq_ids[0] {
        seq_id[0] = 0
    }

    batch.logits[0] = 1
    n_cur += 1

    // Decode the new token
    guard llama_decode(context, batch) == 0 else {
        fatalError("llama_decode failed")
    }
}

let generatedText = String(decoding: generatedBytes, as: UTF8.self)
print("Generated text: \(prompt)\(generatedText)")
```

For more examples,
see the [llama.cpp repo](https://github.com/ggml-org/llama.cpp/tree/master/examples/batched.swift).

## License

This package is available under the MIT license.
See the [LICENSE](LICENSE.md) file for more info.

## Credits

This package wraps the [llama.cpp](https://github.com/ggml-org/llama.cpp) project.
Thanks to all of that project's contributors for making this possible.
