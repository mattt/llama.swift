# llama.swift

A package that provides access to
[llama.cpp](https://github.com/ggml-org/llama.cpp)
in your Swift projects.
It re-exports the llama.cpp C++ APIs directly,
using the precompiled XCFramework provided by the llama.cpp project.

The package automatically stays current with upstream
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases/)
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
- macOS 13.0+ / iOS 16.0+ / tvOS 16.0+ / watchOS 9.0+ / visionOS 1.0+

## Installation

Add this package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/llama.swift", .upToNextMajor(from: "1.7013.0"))
]
```

Or add it through Xcode's Package Manager.

## Usage

```swift
import Llama

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
let utf8Count = prompt.utf8.count

// Create token buffer (simple estimate: UTF-8 count + 1)
let maxTokenCount = utf8Count + 1
var tokens = [llama_token](repeating: 0, count: maxTokenCount)

// Tokenize the input prompt
let tokenCount = llama_tokenize(
    vocab,
    prompt,
    Int32(utf8Count),
    &tokens,
    Int32(maxTokenCount),
    true,  // add BOS (beginning of sequence)
    true   // special tokens
)

guard tokenCount > 0 else {
    fatalError("Failed to tokenize prompt")
}

let promptTokens = Array(tokens.prefix(Int(tokenCount)))

// MARK: - Initial Evaluation

// Create batch for processing
var batch = llama_batch_init(contextParams.n_batch, 0, 1)
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

var generatedText = prompt
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

    // Check for end of sequence
    if nextToken == llama_vocab_eos(vocab) {
        break
    }

    // Convert token to text
    var buffer = [CChar](repeating: 0, count: 16)
    let length = llama_token_to_piece(
        vocab,
        nextToken,
        &buffer,
        Int32(buffer.count),
        0,
        false
    )

    if length > 0 {
        let tokenText = String(cString: buffer)
        generatedText += tokenText
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

print("Generated text: \(generatedText)")
```

For more examples,
see the [llama.cpp repo](https://github.com/ggml-org/llama.cpp/tree/master/examples/batched.swift).

## License

This package is available under the MIT license.
See the [LICENSE](LICENSE.md) file for more info.

## Credits

This package wraps the [llama.cpp](https://github.com/ggml-org/llama.cpp) project.
Thanks to all of that project's contributors for making this possible.
