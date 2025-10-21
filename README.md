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
Llama.backendInit()
defer { Llama.backendFree() }

// Load model
let modelParams = Llama.modelDefaultParams()
guard let model = Llama.modelLoadFromFile("/path/to/model.gguf", modelParams) else {
    fatalError("Failed to load model")
}
defer { Llama.modelFree(model) }

// Create context
var contextParams = Llama.contextDefaultParams()
contextParams.n_ctx = 2048
contextParams.n_batch = 512

guard let context = Llama.initFromModel(model, contextParams) else {
    fatalError("Failed to create context")
}
defer { Llama.free(context) }

// Get vocabulary
let vocab = Llama.modelGetVocab(model)

// Tokenize input
let prompt = "The future of artificial intelligence is"
let cText = prompt.cString(using: .utf8)!
let textLength = Int32(strlen(cText))
var tokenCount = textLength + 1
let tokens = UnsafeMutablePointer<Llama.Token>.allocate(capacity: Int(tokenCount))
defer { tokens.deallocate() }

tokenCount = Llama.tokenize(vocab, prompt, textLength, tokens, tokenCount, true, true)

// Create batch
let batch = Llama.batchGetOne(tokens, tokenCount)
defer { Llama.batchFree(batch) }

// Generate text
var generatedText = prompt
var currentTokens = Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount)))

for _ in 0..<100 { // Generate up to 100 tokens
    // Decode batch
    let result = Llama.decode(context, batch)
    guard result == 0 else { break }
    
    // Get logits and sample next token
    guard let logits = Llama.getLogits(context) else { break }
    
    // Simple greedy sampling (choose highest probability token)
    let vocabSize = Llama.vocabNTokens(vocab)
    var maxLogit = logits[0]
    var nextToken: Llama.Token = 0
    
    for i in 1..<Int(vocabSize) {
        if logits[i] > maxLogit {
            maxLogit = logits[i]
            nextToken = Llama.Token(i)
        }
    }
    
    // Check for end of sequence
    if nextToken == Llama.vocabEos(vocab) { break }
    
    // Convert token to text
    var buffer = [CChar](repeating: 0, count: 16)
    let length = Llama.tokenToPiece(vocab, nextToken, &buffer, Int32(buffer.count), 0, false)
    if length > 0 {
        let tokenText = String(cString: buffer)
        generatedText += tokenText
        print(tokenText, terminator: "")
    }
    
    // Add token to context for next iteration
    currentTokens.append(nextToken)
    let newBatch = Llama.batchGetOne(UnsafeMutablePointer(mutating: currentTokens), Int32(currentTokens.count))
    Llama.batchFree(batch) // Free old batch
    batch = newBatch // Update batch reference
    // Note: batch will be freed by defer at the end
}

print("\n\nGenerated text: \(generatedText)")
```

## License

This package is available under the MIT license. 
See the [LICENSE](LICENSE) file for more info.

## Credits

This package wraps the [llama.cpp](https://github.com/ggerganov/llama.cpp) project by @ggerganov.
