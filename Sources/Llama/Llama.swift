import Foundation
@preconcurrency import llama

/// A Swift wrapper around the llama.cpp C library
///
/// This module provides a clean Swift interface to the llama.cpp library,
/// re-exporting all the necessary types and functions for direct access
/// to the underlying C API.
public enum Llama {
    // MARK: - Types

    /// A token used by the language model
    public typealias Token = llama_token

    /// A batch of tokens for processing
    public typealias Batch = llama_batch

    /// A pointer to the underlying model
    public typealias Model = OpaquePointer

    /// A pointer to the model's context
    public typealias Context = OpaquePointer

    /// A pointer to the model's vocabulary
    public typealias Vocab = OpaquePointer

    /// A pointer to the model's sampler
    public typealias Sampler = UnsafeMutablePointer<llama_sampler>

    /// A pointer to the model's memory
    public typealias Memory = OpaquePointer

    /// A pointer to a LoRA adapter
    public typealias AdapterLora = OpaquePointer

    /// A pointer to a backend
    public typealias Backend = OpaquePointer

    /// A pointer to a backend device
    public typealias BackendDevice = OpaquePointer

    /// A pointer to a backend buffer
    public typealias BackendBuffer = OpaquePointer

    /// A pointer to a backend buffer type
    public typealias BackendBufferType = OpaquePointer

    /// A pointer to a threadpool
    public typealias Threadpool = OpaquePointer

    /// A pointer to a GGML context
    public typealias GGMLContext = OpaquePointer

    /// A pointer to a GGML tensor
    public typealias GGMLTensor = OpaquePointer

    /// A pointer to a GGML computation graph
    public typealias GGMLCGraph = OpaquePointer

    /// Model parameters for loading
    public typealias ModelParams = llama_model_params

    /// Context parameters for initialization
    public typealias ContextParams = llama_context_params

    /// Model quantize parameters
    public typealias ModelQuantizeParams = llama_model_quantize_params

    /// Sampler chain parameters
    public typealias SamplerChainParams = llama_sampler_chain_params

    /// Chat message structure
    public typealias ChatMessage = llama_chat_message

    /// Logit bias structure
    public typealias LogitBias = llama_logit_bias

    /// Token data array
    public typealias TokenDataArray = llama_token_data_array

    /// Performance context data
    public typealias PerfContextData = llama_perf_context_data

    /// Performance sampler data
    public typealias PerfSamplerData = llama_perf_sampler_data

    /// Optimization parameters
    public typealias OptParams = llama_opt_params

    // MARK: - Enums and Constants

    /// Vocabulary types
    public enum VocabType: Int32 {
        case none = 0
        case spm = 1
        case bpe = 2
        case wpm = 3
        case ugm = 4
        case rwkv = 5
    }

    /// RoPE types
    public enum RopeType: Int32 {
        case none = -1
        case norm = 0
        case neox = 2
        case mrope = 8
        case vision = 24
    }

    /// Token types
    public enum TokenType: Int32 {
        case undefined = 0
        case normal = 1
        case unknown = 2
        case control = 3
        case userDefined = 4
        case unused = 5
        case byte = 6
    }

    /// File types
    public enum FType: Int32 {
        case allF32 = 0
        case mostlyF16 = 1
        case mostlyQ4_0 = 2
        case mostlyQ4_1 = 3
        case mostlyQ8_0 = 7
        case mostlyQ5_0 = 8
        case mostlyQ5_1 = 9
        case mostlyQ2_K = 10
        case mostlyQ3_KS = 11
        case mostlyQ3_KM = 12
        case mostlyQ3_KL = 13
        case mostlyQ4_KS = 14
        case mostlyQ4_KM = 15
        case mostlyQ5_KS = 16
        case mostlyQ5_KM = 17
        case mostlyQ6_K = 18
        case mostlyIQ2_XXS = 19
        case mostlyIQ2_XS = 20
        case mostlyQ2_KS = 21
        case mostlyIQ3_XS = 22
        case mostlyIQ3_XXS = 23
        case mostlyIQ1_S = 24
        case mostlyIQ4_NL = 25
        case mostlyIQ3_S = 26
        case mostlyIQ3_M = 27
        case mostlyIQ2_S = 28
        case mostlyIQ2_M = 29
        case mostlyIQ4_XS = 30
        case mostlyIQ1_M = 31
        case mostlyBF16 = 32
        case mostlyTQ1_0 = 36
        case mostlyTQ2_0 = 37
        case guessed = 1024
    }

    /// RoPE scaling types
    public enum RopeScalingType: Int32 {
        case unspecified = -1
        case none = 0
        case linear = 1
        case yarn = 2
        case longrope = 3
    }

    /// Pooling types
    public enum PoolingType: Int32 {
        case unspecified = -1
        case none = 0
        case mean = 1
        case cls = 2
        case last = 3
        case rank = 4
    }

    /// Attention types
    public enum AttentionType: Int32 {
        case unspecified = -1
        case causal = 0
        case nonCausal = 1
    }

    /// Split modes
    public enum SplitMode: Int32 {
        case none = 0
        case layer = 1
        case row = 2
    }

    /// GGML types
    public enum GGMLType: Int32 {
        case f32 = 0
        case f16 = 1
        case q4_0 = 2
        case q4_1 = 3
        case q5_0 = 6
        case q5_1 = 7
        case q8_0 = 8
        case q8_1 = 9
        case q2_K = 10
        case q3_K = 11
        case q4_K = 12
        case q5_K = 13
        case q6_K = 14
        case q8_K = 15
        case iq2_XXS = 16
        case iq2_XS = 17
        case iq3_XXS = 18
        case iq1_S = 19
        case iq4_NL = 20
        case iq3_S = 21
        case iq2_S = 22
        case iq4_XS = 23
        case i8 = 24
        case i16 = 25
        case i32 = 26
        case i64 = 27
        case f64 = 28
        case iq1_M = 29
        case bf16 = 30
        case tq1_0 = 34
        case tq2_0 = 35
    }

    /// GGML log levels
    public enum GGMLLogLevel: Int32 {
        case none = 0
        case debug = 1
        case info = 2
        case warn = 3
        case error = 4
        case cont = 5
    }

    /// GGML operations
    public enum GGMLOp: Int32 {
        case none = 0
        case dup = 1
        case add = 2
        case add1 = 3
        case acc = 4
        case sub = 5
        case mul = 6
        case div = 7
        case sqr = 8
        case sqrt = 9
        case log = 10
        case sin = 11
        case cos = 12
        case sum = 13
        case sumRows = 14
        case mean = 15
        case argmax = 16
        case countEqual = 17
        case `repeat` = 18
        case repeatBack = 19
        case concat = 20
        case siluBack = 21
        case norm = 22
        case rmsNorm = 23
        case rmsNormBack = 24
        case groupNorm = 25
        case l2Norm = 26
        case mulMat = 27
        case mulMatId = 28
        case outProd = 29
        case scale = 30
        case set = 31
        case cpy = 32
        case cont = 33
        case reshape = 34
        case view = 35
        case permute = 36
        case transpose = 37
        case getRows = 38
        case getRowsBack = 39
        case setRows = 40
        case diag = 41
        case diagMaskInf = 42
        case diagMaskZero = 43
        case softMax = 44
        case softMaxBack = 45
        case rope = 46
        case ropeBack = 47
        case clamp = 48
        case convTranspose1D = 49
        case im2col = 50
        case im2colBack = 51
        case conv2D = 52
        case conv2DDw = 53
        case convTranspose2D = 54
        case pool1D = 55
        case pool2D = 56
        case pool2DBack = 57
        case upscale = 58
        case pad = 59
        case padReflect1D = 60
        case roll = 61
        case arange = 62
        case timestepEmbedding = 63
        case argsort = 64
        case leakyRelu = 65
        case flashAttnExt = 66
        case flashAttnBack = 67
        case ssmConv = 68
        case ssmScan = 69
        case winPart = 70
        case winUnpart = 71
        case getRelPos = 72
        case addRelPos = 73
        case rwkvWkv6 = 74
        case gatedLinearAttn = 75
        case rwkvWkv7 = 76
        case unary = 77
        case mapCustom1 = 78
        case mapCustom2 = 79
        case mapCustom3 = 80
        case custom = 81
        case crossEntropyLoss = 82
        case crossEntropyLossBack = 83
        case optStepAdamw = 84
        case glu = 85
    }

    /// GGML unary operations
    public enum GGMLUnaryOp: Int32 {
        case abs = 0
        case sgn = 1
        case neg = 2
        case step = 3
        case tanh = 4
        case elu = 5
        case relu = 6
        case sigmoid = 7
        case gelu = 8
        case geluQuick = 9
        case silu = 10
        case hardswish = 11
        case hardsigmoid = 12
        case exp = 13
        case geluErf = 14
    }

    /// GGML GLU operations
    public enum GGMLGluOp: Int32 {
        case reglu = 0
        case geglu = 1
        case swiglu = 2
        case gegluErf = 3
        case gegluQuick = 4
    }

    /// GGML object types
    public enum GGMLObjectType: Int32 {
        case tensor = 0
        case graph = 1
        case workBuffer = 2
    }

    /// GGML tensor flags
    public enum GGMLTensorFlag: Int32 {
        case input = 1
        case output = 2
        case param = 4
        case loss = 8
    }

    /// GGML precision
    public enum GGMLPrec: Int32 {
        case `default` = 0
        case f32 = 10
    }

    /// GGML status
    public enum GGMLStatus: Int32 {
        case allocFailed = -2
        case failed = -1
        case success = 0
        case aborted = 1
    }

    /// GGML NUMA strategies
    public enum GGMLNumaStrategy: Int32 {
        case disabled = 0
        case distribute = 1
        case isolate = 2
        case numactl = 3
        case mirror = 4
    }

    /// GGML sort orders
    public enum GGMLSortOrder: Int32 {
        case asc = 0
        case desc = 1
    }

    /// GGML scale modes
    public enum GGMLScaleMode: Int32 {
        case nearest = 0
        case bilinear = 1
    }

    /// GGML scale flags
    public enum GGMLScaleFlag: Int32 {
        case alignCorners = 256  // 1 << 8
    }

    /// GGML pool operations
    public enum GGMLOpPool: Int32 {
        case max = 0
        case avg = 1
    }

    /// GGML scheduler priorities
    public enum GGMLSchedPriority: Int32 {
        case low = -1
        case normal = 0
        case medium = 1
        case high = 2
        case realtime = 3
    }

    /// GGML backend device types
    public enum GGMLBackendDeviceType: Int32 {
        case cpu = 0
        case gpu = 1
        case accel = 2
    }

    /// GGML backend buffer usage
    public enum GGMLBackendBufferUsage: Int32 {
        case any = 0
        case weights = 1
        case compute = 2
    }

    // MARK: - Backend and System Functions

    /// Initialize the llama + ggml backend
    public static func backendInit() {
        llama_backend_init()
    }

    /// Free the backend
    public static func backendFree() {
        llama_backend_free()
    }

    /// Initialize NUMA optimizations
    public static func numaInit(_ strategy: GGMLNumaStrategy) {
        llama_numa_init(ggml_numa_strategy(rawValue: UInt32(strategy.rawValue)))
    }

    /// Attach threadpool to context
    public static func attachThreadpool(
        _ context: Context,
        _ threadpool: Threadpool,
        _ threadpoolBatch: Threadpool
    ) {
        llama_attach_threadpool(context, threadpool, threadpoolBatch)
    }

    /// Detach threadpool from context
    public static func detachThreadpool(_ context: Context) {
        llama_detach_threadpool(context)
    }

    /// Get current time in microseconds
    public static func timeUs() -> Int64 {
        return llama_time_us()
    }

    /// Get maximum number of devices
    public static func maxDevices() -> Int {
        return Int(llama_max_devices())
    }

    /// Get maximum number of parallel sequences
    public static func maxParallelSequences() -> Int {
        return Int(llama_max_parallel_sequences())
    }

    /// Check if mmap is supported
    public static func supportsMmap() -> Bool {
        return llama_supports_mmap()
    }

    /// Check if mlock is supported
    public static func supportsMlock() -> Bool {
        return llama_supports_mlock()
    }

    /// Check if GPU offload is supported
    public static func supportsGpuOffload() -> Bool {
        return llama_supports_gpu_offload()
    }

    /// Check if RPC is supported
    public static func supportsRpc() -> Bool {
        return llama_supports_rpc()
    }

    /// Print system information
    public static func printSystemInfo() -> String {
        return String(cString: llama_print_system_info())
    }

    // MARK: - Model Functions

    /// Load a model from file
    public static func modelLoadFromFile(_ path: String, _ params: ModelParams) -> Model? {
        return llama_model_load_from_file(path, params)
    }

    /// Load a model from multiple splits
    public static func modelLoadFromSplits(_ paths: [String], _ params: ModelParams)
        -> Model?
    {
        var cPaths = paths.map { UnsafePointer<CChar>?(strdup($0)) }
        defer {
            cPaths.forEach {
                if let ptr = $0 { Darwin.free(UnsafeMutableRawPointer(mutating: ptr)) }
            }
        }
        return cPaths.withUnsafeMutableBufferPointer { buffer in
            llama_model_load_from_splits(buffer.baseAddress, paths.count, params)
        }
    }

    /// Save model to file
    public static func modelSaveToFile(_ model: Model, _ path: String) {
        llama_model_save_to_file(model, path)
    }

    /// Get default model parameters
    public static func modelDefaultParams() -> ModelParams {
        return llama_model_default_params()
    }

    /// Free a model
    public static func modelFree(_ model: Model) {
        llama_model_free(model)
    }

    /// Get model vocabulary
    public static func modelGetVocab(_ model: Model) -> Vocab {
        return llama_model_get_vocab(model)
    }

    /// Get model RoPE type
    public static func modelRopeType(_ model: Model) -> RopeType {
        return RopeType(rawValue: llama_model_rope_type(model).rawValue)!
    }

    /// Get model context size for training
    public static func modelNCtxTrain(_ model: Model) -> Int32 {
        return llama_model_n_ctx_train(model)
    }

    /// Get model embedding dimension
    public static func modelNEmbd(_ model: Model) -> Int32 {
        return llama_model_n_embd(model)
    }

    /// Get model number of layers
    public static func modelNLayer(_ model: Model) -> Int32 {
        return llama_model_n_layer(model)
    }

    /// Get model number of heads
    public static func modelNHead(_ model: Model) -> Int32 {
        return llama_model_n_head(model)
    }

    /// Get model number of key-value heads
    public static func modelNHeadKv(_ model: Model) -> Int32 {
        return llama_model_n_head_kv(model)
    }

    /// Get model number of SWA
    public static func modelNSwa(_ model: Model) -> Int32 {
        return llama_model_n_swa(model)
    }

    /// Get model RoPE frequency scale for training
    public static func modelRopeFreqScaleTrain(_ model: Model) -> Float {
        return llama_model_rope_freq_scale_train(model)
    }

    /// Get model number of classifier outputs
    public static func modelNClsOut(_ model: Model) -> UInt32 {
        return llama_model_n_cls_out(model)
    }

    /// Get model classifier label
    public static func modelClsLabel(_ model: Model, _ index: UInt32) -> String? {
        guard let label = llama_model_cls_label(model, index) else { return nil }
        return String(cString: label)
    }

    /// Get model metadata value as string
    public static func modelMetaValStr(
        _ model: Model,
        _ key: String,
        _ buffer: UnsafeMutablePointer<CChar>,
        _ bufferSize: Int
    ) -> Int32 {
        return llama_model_meta_val_str(model, key, buffer, bufferSize)
    }

    /// Get model metadata count
    public static func modelMetaCount(_ model: Model) -> Int32 {
        return llama_model_meta_count(model)
    }

    /// Get model metadata key by index
    public static func modelMetaKeyByIndex(
        _ model: Model,
        _ index: Int32,
        _ buffer: UnsafeMutablePointer<CChar>,
        _ bufferSize: Int
    ) -> Int32 {
        return llama_model_meta_key_by_index(model, index, buffer, bufferSize)
    }

    /// Get model metadata value by index
    public static func modelMetaValStrByIndex(
        _ model: Model,
        _ index: Int32,
        _ buffer: UnsafeMutablePointer<CChar>,
        _ bufferSize: Int
    ) -> Int32 {
        return llama_model_meta_val_str_by_index(model, index, buffer, bufferSize)
    }

    /// Get model description
    public static func modelDesc(
        _ model: Model,
        _ buffer: UnsafeMutablePointer<CChar>,
        _ bufferSize: Int
    ) -> Int32 {
        return llama_model_desc(model, buffer, bufferSize)
    }

    /// Get model size in bytes
    public static func modelSize(_ model: Model) -> UInt64 {
        return llama_model_size(model)
    }

    /// Get model chat template
    public static func modelChatTemplate(_ model: Model, _ name: String?) -> String? {
        guard let template = llama_model_chat_template(model, name) else { return nil }
        return String(cString: template)
    }

    /// Get model number of parameters
    public static func modelNParams(_ model: Model) -> UInt64 {
        return llama_model_n_params(model)
    }

    /// Check if model has encoder
    public static func modelHasEncoder(_ model: Model) -> Bool {
        return llama_model_has_encoder(model)
    }

    /// Check if model has decoder
    public static func modelHasDecoder(_ model: Model) -> Bool {
        return llama_model_has_decoder(model)
    }

    /// Get model decoder start token
    public static func modelDecoderStartToken(_ model: Model) -> Token {
        return llama_model_decoder_start_token(model)
    }

    /// Check if model is recurrent
    public static func modelIsRecurrent(_ model: Model) -> Bool {
        return llama_model_is_recurrent(model)
    }

    /// Quantize model
    public static func modelQuantize(
        _ fnameInp: String,
        _ fnameOut: String,
        _ params: ModelQuantizeParams
    ) -> UInt32 {
        var mutableParams = params
        return llama_model_quantize(fnameInp, fnameOut, &mutableParams)
    }

    /// Get default model quantize parameters
    public static func modelQuantizeDefaultParams() -> ModelQuantizeParams {
        return llama_model_quantize_default_params()
    }

    // MARK: - Context Functions

    /// Initialize context from model
    public static func initFromModel(_ model: Model, _ params: ContextParams) -> Context? {
        return llama_init_from_model(model, params)
    }

    /// Get default context parameters
    public static func contextDefaultParams() -> ContextParams {
        return llama_context_default_params()
    }

    /// Free context
    public static func free(_ context: Context) {
        llama_free(context)
    }

    /// Get context size
    public static func nCtx(_ context: Context) -> UInt32 {
        return llama_n_ctx(context)
    }

    /// Get context batch size
    public static func nBatch(_ context: Context) -> UInt32 {
        return llama_n_batch(context)
    }

    /// Get context ubatch size
    public static func nUbatch(_ context: Context) -> UInt32 {
        return llama_n_ubatch(context)
    }

    /// Get context max sequences
    public static func nSeqMax(_ context: Context) -> UInt32 {
        return llama_n_seq_max(context)
    }

    /// Get model from context
    public static func getModel(_ context: Context) -> Model {
        return llama_get_model(context)
    }

    /// Get memory from context
    public static func getMemory(_ context: Context) -> Memory {
        return llama_get_memory(context)
    }

    /// Get pooling type from context
    public static func poolingType(_ context: Context) -> PoolingType {
        return PoolingType(rawValue: llama_pooling_type(context).rawValue)!
    }

    /// Set number of threads
    public static func setNThreads(_ context: Context, _ nThreads: Int32, _ nThreadsBatch: Int32) {
        llama_set_n_threads(context, nThreads, nThreadsBatch)
    }

    /// Get number of threads
    public static func nThreads(_ context: Context) -> Int32 {
        return llama_n_threads(context)
    }

    /// Get number of batch threads
    public static func nThreadsBatch(_ context: Context) -> Int32 {
        return llama_n_threads_batch(context)
    }

    /// Set embeddings mode
    public static func setEmbeddings(_ context: Context, _ embeddings: Bool) {
        llama_set_embeddings(context, embeddings)
    }

    /// Set causal attention
    public static func setCausalAttn(_ context: Context, _ causalAttn: Bool) {
        llama_set_causal_attn(context, causalAttn)
    }

    /// Set warmup mode
    public static func setWarmup(_ context: Context, _ warmup: Bool) {
        llama_set_warmup(context, warmup)
    }

    /// Set abort callback
    public static func setAbortCallback(
        _ context: Context,
        _ abortCallback: @escaping @convention(c) (UnsafeMutableRawPointer?) -> Bool,
        _ abortCallbackData: UnsafeMutableRawPointer?
    ) {
        llama_set_abort_callback(context, abortCallback, abortCallbackData)
    }

    /// Synchronize context
    public static func synchronize(_ context: Context) {
        llama_synchronize(context)
    }

    // MARK: - Decoding Functions

    /// Get batch for single sequence
    public static func batchGetOne(_ tokens: UnsafeMutablePointer<Token>, _ nTokens: Int32) -> Batch {
        return llama_batch_get_one(tokens, nTokens)
    }

    /// Initialize batch
    public static func batchInit(_ nTokens: Int32, _ embd: Int32, _ nSeqMax: Int32) -> Batch {
        return llama_batch_init(nTokens, embd, nSeqMax)
    }

    /// Free batch
    public static func batchFree(_ batch: Batch) {
        llama_batch_free(batch)
    }

    /// Encode batch
    public static func encode(_ context: Context, _ batch: Batch) -> Int32 {
        return llama_encode(context, batch)
    }

    /// Decode batch
    public static func decode(_ context: Context, _ batch: Batch) -> Int32 {
        return llama_decode(context, batch)
    }

    /// Get logits
    public static func getLogits(_ context: Context) -> UnsafeMutablePointer<Float>? {
        return llama_get_logits(context)
    }

    /// Get logits for specific token
    public static func getLogitsIth(_ context: Context, _ i: Int32) -> UnsafeMutablePointer<Float>? {
        return llama_get_logits_ith(context, i)
    }

    /// Get embeddings
    public static func getEmbeddings(_ context: Context) -> UnsafeMutablePointer<Float>? {
        return llama_get_embeddings(context)
    }

    /// Get embeddings for specific token
    public static func getEmbeddingsIth(_ context: Context, _ i: Int32) -> UnsafeMutablePointer<
        Float
    >? {
        return llama_get_embeddings_ith(context, i)
    }

    /// Get embeddings for sequence
    public static func getEmbeddingsSeq(_ context: Context, _ seqId: Int32) -> UnsafeMutablePointer<
        Float
    >? {
        return llama_get_embeddings_seq(context, seqId)
    }

    // MARK: - Vocabulary Functions

    /// Get vocabulary type
    public static func vocabType(_ vocab: Vocab) -> VocabType {
        return VocabType(rawValue: Int32(llama_vocab_type(vocab).rawValue))!
    }

    /// Get vocabulary token count
    public static func vocabNTokens(_ vocab: Vocab) -> Int32 {
        return llama_vocab_n_tokens(vocab)
    }

    /// Get vocabulary text for token
    public static func vocabGetText(_ vocab: Vocab, _ token: Token) -> String? {
        guard let text = llama_vocab_get_text(vocab, token) else { return nil }
        return String(cString: text)
    }

    /// Get vocabulary score for token
    public static func vocabGetScore(_ vocab: Vocab, _ token: Token) -> Float {
        return llama_vocab_get_score(vocab, token)
    }

    /// Get vocabulary attributes for token
    public static func vocabGetAttr(_ vocab: Vocab, _ token: Token) -> UInt32 {
        return llama_vocab_get_attr(vocab, token).rawValue
    }

    /// Check if token is end of generation
    public static func vocabIsEog(_ vocab: Vocab, _ token: Token) -> Bool {
        return llama_vocab_is_eog(vocab, token)
    }

    /// Check if token is control
    public static func vocabIsControl(_ vocab: Vocab, _ token: Token) -> Bool {
        return llama_vocab_is_control(vocab, token)
    }

    /// Get vocabulary BOS token
    public static func vocabBos(_ vocab: Vocab) -> Token {
        return llama_vocab_bos(vocab)
    }

    /// Get vocabulary EOS token
    public static func vocabEos(_ vocab: Vocab) -> Token {
        return llama_vocab_eos(vocab)
    }

    /// Get vocabulary EOT token
    public static func vocabEot(_ vocab: Vocab) -> Token {
        return llama_vocab_eot(vocab)
    }

    /// Get vocabulary SEP token
    public static func vocabSep(_ vocab: Vocab) -> Token {
        return llama_vocab_sep(vocab)
    }

    /// Get vocabulary NL token
    public static func vocabNl(_ vocab: Vocab) -> Token {
        return llama_vocab_nl(vocab)
    }

    /// Get vocabulary PAD token
    public static func vocabPad(_ vocab: Vocab) -> Token {
        return llama_vocab_pad(vocab)
    }

    /// Get vocabulary add BOS flag
    public static func vocabGetAddBos(_ vocab: Vocab) -> Bool {
        return llama_vocab_get_add_bos(vocab)
    }

    /// Get vocabulary add EOS flag
    public static func vocabGetAddEos(_ vocab: Vocab) -> Bool {
        return llama_vocab_get_add_eos(vocab)
    }

    /// Get vocabulary add SEP flag
    public static func vocabGetAddSep(_ vocab: Vocab) -> Bool {
        return llama_vocab_get_add_sep(vocab)
    }

    /// Get vocabulary FIM PRE token
    public static func vocabFimPre(_ vocab: Vocab) -> Token {
        return llama_vocab_fim_pre(vocab)
    }

    /// Get vocabulary FIM SUF token
    public static func vocabFimSuf(_ vocab: Vocab) -> Token {
        return llama_vocab_fim_suf(vocab)
    }

    /// Get vocabulary FIM MID token
    public static func vocabFimMid(_ vocab: Vocab) -> Token {
        return llama_vocab_fim_mid(vocab)
    }

    /// Get vocabulary FIM PAD token
    public static func vocabFimPad(_ vocab: Vocab) -> Token {
        return llama_vocab_fim_pad(vocab)
    }

    /// Get vocabulary FIM REP token
    public static func vocabFimRep(_ vocab: Vocab) -> Token {
        return llama_vocab_fim_rep(vocab)
    }

    /// Get vocabulary FIM SEP token
    public static func vocabFimSep(_ vocab: Vocab) -> Token {
        return llama_vocab_fim_sep(vocab)
    }

    // MARK: - Tokenization Functions

    /// Tokenize text
    public static func tokenize(
        _ vocab: Vocab,
        _ text: String,
        _ textLength: Int32,
        _ tokens: UnsafeMutablePointer<Token>,
        _ maxTokens: Int32,
        _ addSpecial: Bool,
        _ parseSpecial: Bool
    ) -> Int32 {
        return llama_tokenize(vocab, text, textLength, tokens, maxTokens, addSpecial, parseSpecial)
    }

    /// Convert token to piece
    public static func tokenToPiece(
        _ vocab: Vocab,
        _ token: Token,
        _ buffer: UnsafeMutablePointer<CChar>,
        _ length: Int32,
        _ lstrip: Int32,
        _ special: Bool
    ) -> Int32 {
        return llama_token_to_piece(vocab, token, buffer, length, lstrip, special)
    }

    /// Detokenize tokens to text
    public static func detokenize(
        _ vocab: Vocab,
        _ tokens: UnsafePointer<Token>,
        _ nTokens: Int32,
        _ text: UnsafeMutablePointer<CChar>,
        _ textLengthMax: Int32,
        _ removeSpecial: Bool,
        _ unparseSpecial: Bool
    ) -> Int32 {
        return llama_detokenize(
            vocab,
            tokens,
            nTokens,
            text,
            textLengthMax,
            removeSpecial,
            unparseSpecial
        )
    }

    // MARK: - Chat Template Functions

    /// Apply chat template
    public static func chatApplyTemplate(
        _ tmpl: String?,
        _ chat: UnsafePointer<ChatMessage>,
        _ nMsg: Int,
        _ addAss: Bool,
        _ buffer: UnsafeMutablePointer<CChar>,
        _ length: Int32
    ) -> Int32 {
        return llama_chat_apply_template(tmpl, chat, nMsg, addAss, buffer, length)
    }

    /// Get built-in chat templates
    public static func chatBuiltinTemplates(
        _ output: UnsafeMutablePointer<UnsafePointer<CChar>?>,
        _ len: Int
    ) -> Int32 {
        return llama_chat_builtin_templates(output, len)
    }

    // MARK: - Memory Functions

    /// Clear memory
    public static func memoryClear(_ memory: Memory, _ data: Bool) {
        llama_memory_clear(memory, data)
    }

    /// Remove memory sequence
    public static func memorySeqRm(_ memory: Memory, _ seqId: Int32, _ p0: Int32, _ p1: Int32)
        -> Bool
    {
        return llama_memory_seq_rm(memory, seqId, p0, p1)
    }

    /// Copy memory sequence
    public static func memorySeqCp(
        _ memory: Memory,
        _ seqIdSrc: Int32,
        _ seqIdDst: Int32,
        _ p0: Int32,
        _ p1: Int32
    ) {
        llama_memory_seq_cp(memory, seqIdSrc, seqIdDst, p0, p1)
    }

    /// Keep memory sequence
    public static func memorySeqKeep(_ memory: Memory, _ seqId: Int32) {
        llama_memory_seq_keep(memory, seqId)
    }

    /// Add to memory sequence
    public static func memorySeqAdd(
        _ memory: Memory,
        _ seqId: Int32,
        _ p0: Int32,
        _ p1: Int32,
        _ delta: Int32
    ) {
        llama_memory_seq_add(memory, seqId, p0, p1, delta)
    }

    /// Divide memory sequence
    public static func memorySeqDiv(
        _ memory: Memory,
        _ seqId: Int32,
        _ p0: Int32,
        _ p1: Int32,
        _ d: Int32
    ) {
        llama_memory_seq_div(memory, seqId, p0, p1, d)
    }

    /// Get memory sequence minimum position
    public static func memorySeqPosMin(_ memory: Memory, _ seqId: Int32) -> Int32 {
        return llama_memory_seq_pos_min(memory, seqId)
    }

    /// Get memory sequence maximum position
    public static func memorySeqPosMax(_ memory: Memory, _ seqId: Int32) -> Int32 {
        return llama_memory_seq_pos_max(memory, seqId)
    }

    /// Check if memory can shift
    public static func memoryCanShift(_ memory: Memory) -> Bool {
        return llama_memory_can_shift(memory)
    }

    // MARK: - State Functions

    /// Get state size
    public static func stateGetSize(_ context: Context) -> Int {
        return Int(llama_state_get_size(context))
    }

    /// Get state data
    public static func stateGetData(
        _ context: Context,
        _ dst: UnsafeMutablePointer<UInt8>,
        _ size: Int
    ) -> Int {
        return Int(llama_state_get_data(context, dst, size))
    }

    /// Set state data
    public static func stateSetData(_ context: Context, _ src: UnsafePointer<UInt8>, _ size: Int)
        -> Int
    {
        return Int(llama_state_set_data(context, src, size))
    }

    /// Load state from file
    public static func stateLoadFile(
        _ context: Context,
        _ pathSession: String,
        _ tokensOut: UnsafeMutablePointer<Token>,
        _ nTokenCapacity: Int,
        _ nTokenCountOut: UnsafeMutablePointer<Int>
    ) -> Bool {
        return llama_state_load_file(
            context,
            pathSession,
            tokensOut,
            nTokenCapacity,
            nTokenCountOut
        )
    }

    /// Save state to file
    public static func stateSaveFile(
        _ context: Context,
        _ pathSession: String,
        _ tokens: UnsafePointer<Token>,
        _ nTokenCount: Int
    ) -> Bool {
        return llama_state_save_file(context, pathSession, tokens, nTokenCount)
    }

    /// Get state sequence size
    public static func stateSeqGetSize(_ context: Context, _ seqId: Int32) -> Int {
        return Int(llama_state_seq_get_size(context, seqId))
    }

    /// Get state sequence data
    public static func stateSeqGetData(
        _ context: Context,
        _ dst: UnsafeMutablePointer<UInt8>,
        _ size: Int,
        _ seqId: Int32
    ) -> Int {
        return Int(llama_state_seq_get_data(context, dst, size, seqId))
    }

    /// Set state sequence data
    public static func stateSeqSetData(
        _ context: Context,
        _ src: UnsafePointer<UInt8>,
        _ size: Int,
        _ destSeqId: Int32
    ) -> Int {
        return Int(llama_state_seq_set_data(context, src, size, destSeqId))
    }

    /// Save state sequence to file
    public static func stateSeqSaveFile(
        _ context: Context,
        _ filepath: String,
        _ seqId: Int32,
        _ tokens: UnsafePointer<Token>,
        _ nTokenCount: Int
    ) -> Int {
        return Int(llama_state_seq_save_file(context, filepath, seqId, tokens, nTokenCount))
    }

    /// Load state sequence from file
    public static func stateSeqLoadFile(
        _ context: Context,
        _ filepath: String,
        _ destSeqId: Int32,
        _ tokensOut: UnsafeMutablePointer<Token>,
        _ nTokenCapacity: Int,
        _ nTokenCountOut: UnsafeMutablePointer<Int>
    ) -> Int {
        return Int(
            llama_state_seq_load_file(
                context,
                filepath,
                destSeqId,
                tokensOut,
                nTokenCapacity,
                nTokenCountOut
            )
        )
    }

    // MARK: - Adapter Functions

    /// Initialize LoRA adapter
    public static func adapterLoraInit(_ model: Model, _ pathLora: String) -> AdapterLora? {
        return llama_adapter_lora_init(model, pathLora)
    }

    /// Free LoRA adapter
    public static func adapterLoraFree(_ adapter: AdapterLora) {
        llama_adapter_lora_free(adapter)
    }

    /// Set LoRA adapter
    public static func setAdapterLora(_ context: Context, _ adapter: AdapterLora, _ scale: Float)
        -> Int32
    {
        return llama_set_adapter_lora(context, adapter, scale)
    }

    /// Remove LoRA adapter
    public static func rmAdapterLora(_ context: Context, _ adapter: AdapterLora) -> Int32 {
        return llama_rm_adapter_lora(context, adapter)
    }

    /// Clear LoRA adapters
    public static func clearAdapterLora(_ context: Context) {
        llama_clear_adapter_lora(context)
    }

    /// Apply adapter control vector
    public static func applyAdapterCvec(
        _ context: Context,
        _ data: UnsafePointer<Float>,
        _ len: Int,
        _ nEmbd: Int32,
        _ ilStart: Int32,
        _ ilEnd: Int32
    ) -> Int32 {
        return llama_apply_adapter_cvec(context, data, len, nEmbd, ilStart, ilEnd)
    }

    // MARK: - Sampling Functions

    /// Initialize sampler chain
    public static func samplerChainInit(_ params: SamplerChainParams) -> Sampler? {
        return llama_sampler_chain_init(params)
    }

    /// Get default sampler chain parameters
    public static func samplerChainDefaultParams() -> SamplerChainParams {
        return llama_sampler_chain_default_params()
    }

    /// Add sampler to chain
    public static func samplerChainAdd(
        _ sampler: Sampler,
        _ component: UnsafeMutablePointer<llama_sampler>
    ) {
        llama_sampler_chain_add(sampler, component)
    }

    /// Get sampler from chain
    public static func samplerChainGet(_ chain: Sampler, _ i: Int32) -> UnsafeMutablePointer<
        llama_sampler
    >? {
        return llama_sampler_chain_get(chain, i)
    }

    /// Get sampler chain count
    public static func samplerChainN(_ chain: Sampler) -> Int32 {
        return llama_sampler_chain_n(chain)
    }

    /// Remove sampler from chain
    public static func samplerChainRemove(_ chain: Sampler, _ i: Int32) -> UnsafeMutablePointer<
        llama_sampler
    >? {
        return llama_sampler_chain_remove(chain, i)
    }

    /// Initialize greedy sampler
    public static func samplerInitGreedy() -> UnsafeMutablePointer<llama_sampler> {
        return llama_sampler_init_greedy()
    }

    /// Initialize distribution sampler
    public static func samplerInitDist(_ seed: UInt32) -> UnsafeMutablePointer<llama_sampler> {
        return llama_sampler_init_dist(seed)
    }

    /// Initialize top-k sampler
    public static func samplerInitTopK(_ k: Int32) -> UnsafeMutablePointer<llama_sampler> {
        return llama_sampler_init_top_k(k)
    }

    /// Initialize top-p sampler
    public static func samplerInitTopP(_ p: Float, _ minKeep: Int) -> UnsafeMutablePointer<
        llama_sampler
    > {
        return llama_sampler_init_top_p(p, minKeep)
    }

    /// Initialize min-p sampler
    public static func samplerInitMinP(_ p: Float, _ minKeep: Int) -> UnsafeMutablePointer<
        llama_sampler
    > {
        return llama_sampler_init_min_p(p, minKeep)
    }

    /// Initialize typical sampler
    public static func samplerInitTypical(_ p: Float, _ minKeep: Int) -> UnsafeMutablePointer<
        llama_sampler
    > {
        return llama_sampler_init_typical(p, minKeep)
    }

    /// Initialize temperature sampler
    public static func samplerInitTemp(_ temp: Float) -> UnsafeMutablePointer<llama_sampler> {
        return llama_sampler_init_temp(temp)
    }

    /// Initialize temperature extended sampler
    public static func samplerInitTempExt(_ temp: Float, _ delta: Float, _ exponent: Float)
        -> UnsafeMutablePointer<llama_sampler>
    {
        return llama_sampler_init_temp_ext(temp, delta, exponent)
    }

    /// Initialize XTC sampler
    public static func samplerInitXtc(_ p: Float, _ temp: Float, _ minKeep: Int, _ seed: UInt32)
        -> UnsafeMutablePointer<llama_sampler>
    {
        return llama_sampler_init_xtc(p, temp, minKeep, seed)
    }

    /// Initialize top-n sigma sampler
    public static func samplerInitTopNSigma(_ n: Float) -> UnsafeMutablePointer<llama_sampler> {
        return llama_sampler_init_top_n_sigma(n)
    }

    /// Initialize Mirostat sampler
    public static func samplerInitMirostat(
        _ nVocab: Int32,
        _ seed: UInt32,
        _ tau: Float,
        _ eta: Float,
        _ m: Int32
    ) -> UnsafeMutablePointer<llama_sampler> {
        return llama_sampler_init_mirostat(nVocab, seed, tau, eta, m)
    }

    /// Initialize Mirostat v2 sampler
    public static func samplerInitMirostatV2(_ seed: UInt32, _ tau: Float, _ eta: Float)
        -> UnsafeMutablePointer<llama_sampler>
    {
        return llama_sampler_init_mirostat_v2(seed, tau, eta)
    }

    /// Initialize grammar sampler
    public static func samplerInitGrammar(
        _ vocab: Vocab,
        _ grammarStr: String,
        _ grammarRoot: String
    ) -> UnsafeMutablePointer<llama_sampler>? {
        return llama_sampler_init_grammar(vocab, grammarStr, grammarRoot)
    }

    /// Initialize lazy grammar sampler with patterns
    public static func samplerInitGrammarLazyPatterns(
        _ vocab: Vocab,
        _ grammarStr: String,
        _ grammarRoot: String,
        _ triggerPatterns: UnsafeMutablePointer<UnsafePointer<CChar>?>,
        _ numTriggerPatterns: Int,
        _ triggerTokens: UnsafePointer<Token>,
        _ numTriggerTokens: Int
    ) -> UnsafeMutablePointer<llama_sampler>? {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammarStr,
            grammarRoot,
            triggerPatterns,
            numTriggerPatterns,
            triggerTokens,
            numTriggerTokens
        )
    }

    /// Initialize penalties sampler
    public static func samplerInitPenalties(
        _ penaltyLastN: Int32,
        _ penaltyRepeat: Float,
        _ penaltyFreq: Float,
        _ penaltyPresent: Float
    ) -> UnsafeMutablePointer<llama_sampler> {
        return llama_sampler_init_penalties(
            penaltyLastN,
            penaltyRepeat,
            penaltyFreq,
            penaltyPresent
        )
    }

    /// Initialize DRY sampler
    public static func samplerInitDry(
        _ vocab: Vocab,
        _ nCtxTrain: Int32,
        _ dryMultiplier: Float,
        _ dryBase: Float,
        _ dryAllowedLength: Int32,
        _ dryPenaltyLastN: Int32,
        _ seqBreakers: UnsafeMutablePointer<UnsafePointer<CChar>?>,
        _ numBreakers: Int
    ) -> UnsafeMutablePointer<llama_sampler> {
        return llama_sampler_init_dry(
            vocab,
            nCtxTrain,
            dryMultiplier,
            dryBase,
            dryAllowedLength,
            dryPenaltyLastN,
            seqBreakers,
            numBreakers
        )
    }

    /// Initialize logit bias sampler
    public static func samplerInitLogitBias(
        _ nVocab: Int32,
        _ nLogitBias: Int32,
        _ logitBias: UnsafePointer<LogitBias>
    ) -> UnsafeMutablePointer<llama_sampler> {
        return llama_sampler_init_logit_bias(nVocab, nLogitBias, logitBias)
    }

    /// Initialize infill sampler
    public static func samplerInitInfill(_ vocab: Vocab) -> UnsafeMutablePointer<llama_sampler> {
        return llama_sampler_init_infill(vocab)
    }

    /// Get sampler seed
    public static func samplerGetSeed(_ sampler: UnsafeMutablePointer<llama_sampler>) -> UInt32 {
        return llama_sampler_get_seed(sampler)
    }

    /// Sample token
    public static func samplerSample(
        _ sampler: UnsafeMutablePointer<llama_sampler>,
        _ context: Context,
        _ idx: Int32
    ) -> Token {
        return llama_sampler_sample(sampler, context, idx)
    }

    /// Accept token
    public static func samplerAccept(_ sampler: UnsafeMutablePointer<llama_sampler>, _ token: Token) {
        llama_sampler_accept(sampler, token)
    }

    /// Apply sampler
    public static func samplerApply(
        _ sampler: UnsafeMutablePointer<llama_sampler>,
        _ curP: UnsafeMutablePointer<TokenDataArray>
    ) {
        llama_sampler_apply(sampler, curP)
    }

    /// Reset sampler
    public static func samplerReset(_ sampler: UnsafeMutablePointer<llama_sampler>) {
        llama_sampler_reset(sampler)
    }

    /// Clone sampler
    public static func samplerClone(_ sampler: UnsafeMutablePointer<llama_sampler>)
        -> UnsafeMutablePointer<llama_sampler>?
    {
        return llama_sampler_clone(sampler)
    }

    /// Free sampler
    public static func samplerFree(_ sampler: UnsafeMutablePointer<llama_sampler>) {
        llama_sampler_free(sampler)
    }

    /// Get sampler name
    public static func samplerName(_ sampler: UnsafeMutablePointer<llama_sampler>) -> String? {
        guard let name = llama_sampler_name(sampler) else { return nil }
        return String(cString: name)
    }

    // MARK: - Model Split Functions

    /// Build split path
    public static func splitPath(
        _ splitPath: UnsafeMutablePointer<CChar>,
        _ maxlen: Int,
        _ pathPrefix: String,
        _ splitNo: Int,
        _ splitCount: Int
    ) -> Int {
        return Int(
            llama_split_path(splitPath, maxlen, pathPrefix, Int32(splitNo), Int32(splitCount))
        )
    }

    /// Extract split prefix
    public static func splitPrefix(
        _ splitPrefix: UnsafeMutablePointer<CChar>,
        _ maxlen: Int,
        _ splitPath: String,
        _ splitNo: Int,
        _ splitCount: Int
    ) -> Int {
        return Int(
            llama_split_prefix(splitPrefix, maxlen, splitPath, Int32(splitNo), Int32(splitCount))
        )
    }

    // MARK: - Logging Functions

    /// Set log callback
    public static func logSet(
        _ callback:
            @escaping @convention(c) (
                ggml_log_level, UnsafePointer<CChar>?, UnsafeMutableRawPointer?
            ) -> Void,
        _ userData: UnsafeMutableRawPointer?
    ) {
        llama_log_set(callback, userData)
    }

    /// Set GGML log callback
    public static func ggmlLogSet(
        _ callback:
            @escaping @convention(c) (
                ggml_log_level, UnsafePointer<CChar>?, UnsafeMutableRawPointer?
            ) -> Void,
        _ userData: UnsafeMutableRawPointer?
    ) {
        ggml_log_set(callback, userData)
    }

    // MARK: - Performance Functions

    /// Get context performance data
    public static func perfContext(_ context: Context) -> PerfContextData {
        return llama_perf_context(context)
    }

    /// Print context performance
    public static func perfContextPrint(_ context: Context) {
        llama_perf_context_print(context)
    }

    /// Reset context performance
    public static func perfContextReset(_ context: Context) {
        llama_perf_context_reset(context)
    }

    /// Get sampler performance data
    public static func perfSampler(_ chain: UnsafeMutablePointer<llama_sampler>)
        -> PerfSamplerData
    {
        return llama_perf_sampler(chain)
    }

    /// Print sampler performance
    public static func perfSamplerPrint(_ chain: UnsafeMutablePointer<llama_sampler>) {
        llama_perf_sampler_print(chain)
    }

    /// Reset sampler performance
    public static func perfSamplerReset(_ chain: UnsafeMutablePointer<llama_sampler>) {
        llama_perf_sampler_reset(chain)
    }

    // MARK: - Training Functions

    /// Initialize optimization
    public static func optInit(_ context: Context, _ model: Model, _ loptParams: OptParams) {
        llama_opt_init(context, model, loptParams)
    }

    /// Optimization epoch
    public static func optEpoch(
        _ context: Context,
        _ dataset: ggml_opt_dataset_t,
        _ resultTrain: ggml_opt_result_t,
        _ resultEval: ggml_opt_result_t,
        _ idataSplit: Int64,
        _ callbackTrain: ggml_opt_epoch_callback,
        _ callbackEval: ggml_opt_epoch_callback
    ) {
        llama_opt_epoch(
            context,
            dataset,
            resultTrain,
            resultEval,
            idataSplit,
            callbackTrain,
            callbackEval
        )
    }

    /// Parameter filter for all
    public static func optParamFilterAll(
        _ tensor: UnsafeMutablePointer<ggml_tensor>,
        _ userdata: UnsafeMutableRawPointer?
    ) -> Bool {
        return llama_opt_param_filter_all(tensor, userdata)
    }
}
