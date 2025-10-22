# llama-swift

A Swift wrapper for [llama.cpp](https://github.com/ggerganov/llama.cpp)
that provides direct access to the underlying C API through modern Swift 6 C++ interoperability.
Use it directly, or build your own abstraction on top of it.

## Requirements

- Swift 6.0+
- macOS 13.0+ / iOS 16.0+ / tvOS 16.0+ / watchOS 9.0+ / visionOS 1.0+

## Installation

Add this package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/llama-swift.git", exact: "1.6816.0")
]
```

Or add it through Xcode's Package Manager.

## Usage

```swift
import Llama

// Initialize the backend
llama_backend_init()
defer { llama_backend_free() }

// Load model
let modelParams = llama_model_default_params()
guard let model = llama_model_load_from_file("/path/to/model.gguf", modelParams) else {
    fatalError("Failed to load model")
}
defer { llama_model_free(model) }

// Create context
var contextParams = llama_context_default_params()
contextParams.n_ctx = 2048
contextParams.n_batch = 512

guard let context = llama_init_from_model(model, contextParams) else {
    fatalError("Failed to create context")
}
defer { llama_free(context) }

// Get vocabulary
let vocab = llama_model_get_vocab(model)

// Tokenize input
let prompt = "The future of artificial intelligence is"
let utf8Count = prompt.utf8.count
// A buffer large enough to hold all tokens. A simple estimate.
let maxTokenCount = utf8Count + 1
var tokens = [llama_token](repeating: 0, count: maxTokenCount)
let tokenCount = llama_tokenize(vocab, prompt, Int32(utf8Count), &tokens, Int32(maxTokenCount), /* add bos */ true, /* special */ true)
guard tokenCount > 0 else {
    fatalError("Failed to tokenize prompt")
}
let promptTokens = Array(tokens.prefix(Int(tokenCount)))

// Create a batch for processing
var batch = llama_batch_init(contextParams.n_batch, 0, 1)
defer { llama_batch_free(batch) }

// Evaluate the prompt
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
// Only get logits for the last token
if batch.n_tokens > 0 {
    batch.logits[Int(batch.n_tokens) - 1] = 1
}

guard llama_decode(context, batch) == 0 else {
    fatalError("llama_decode failed")
}

// Generate text
var generatedText = prompt
var n_cur = batch.n_tokens

for _ in 0..<100 { // Generate up to 100 tokens
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
    if nextToken == llama_vocab_eos(vocab) { break }

    // Convert token to text and print it
    var buffer = [CChar](repeating: 0, count: 16)
    let length = llama_token_to_piece(vocab, nextToken, &buffer, Int32(buffer.count), 0, false)
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

print("\n\nGenerated text: \(generatedText)")
```

## License

This package is available under the MIT license.
See the [LICENSE](LICENSE) file for more info.

## Credits

This package wraps the [llama.cpp](https://github.com/ggerganov/llama.cpp) project by @ggerganov.
