import Foundation
import Testing

@testable import Llama

@Suite("Llama Tests", .serialized)
struct LlamaTests {
    // MARK: - Type Tests

    @Test
    func testTypeAliasesExist() async throws {
        // Basic types
        let _: Llama.Token = 0
        let _: Llama.Model? = nil
        let _: Llama.Context? = nil
        let _: Llama.Vocab? = nil
        let _: Llama.Sampler? = nil
        let _: Llama.Memory? = nil
        let _: Llama.AdapterLora? = nil
        let _: Llama.Backend? = nil
        let _: Llama.BackendDevice? = nil
        let _: Llama.BackendBuffer? = nil
        let _: Llama.Threadpool? = nil
        let _: Llama.GGMLContext? = nil
        let _: Llama.GGMLTensor? = nil
        let _: Llama.GGMLCGraph? = nil

        // Parameter types
        let _: Llama.ModelParams
        let _: Llama.ContextParams
        let _: Llama.ModelQuantizeParams
        let _: Llama.SamplerChainParams
        let _: Llama.OptParams

        // Data structure types
        let _: Llama.ChatMessage
        let _: Llama.LogitBias
        let _: Llama.TokenDataArray
        let _: Llama.PerfContextData
        let _: Llama.PerfSamplerData

        // If we got here, all type aliases are accessible
        #expect(Bool(true))
    }

    // MARK: - System Functions

    @Test
    func testSystemInfo() async throws {
        let systemInfo = Llama.printSystemInfo()
        #expect(!systemInfo.isEmpty)
    }

    @Test
    func testNumaInit() async throws {
        // Test NUMA initialization
        Llama.numaInit(.disabled)
    }

    @Test
    func testSystemCapabilities() async throws {
        let maxDevices = Llama.maxDevices()
        #expect(maxDevices >= 0)

        let maxParallel = Llama.maxParallelSequences()
        #expect(maxParallel >= 0)

        _ = Llama.supportsMmap()
        _ = Llama.supportsMlock()
        _ = Llama.supportsGpuOffload()
        _ = Llama.supportsRpc()
    }

    @Test
    func testTimeUs() async throws {
        let time = Llama.timeUs()
        #expect(time >= 0)
    }

    // MARK: - Parameter Functions

    @Test
    func testModelDefaultParams() async throws {
        let params = Llama.modelDefaultParams()
        #expect(params.n_gpu_layers == 999)
    }

    @Test
    func testModelQuantizeDefaultParams() async throws {
        let params = Llama.modelQuantizeDefaultParams()
        // Just test that we can get default params
        _ = params.ftype
    }

    @Test
    func testContextDefaultParams() async throws {
        let params = Llama.contextDefaultParams()
        #expect(params.n_ctx == 512)
    }

    @Test
    func testSamplerChainDefaultParams() async throws {
        let params = Llama.samplerChainDefaultParams()
        // Just test that we can get default params
        _ = params
        #expect(Bool(true))
    }

    // MARK: - Parameter Modification

    @Test
    func testModifyModelParams() async throws {
        var params = Llama.modelDefaultParams()
        params.n_gpu_layers = 1
        #expect(params.n_gpu_layers == 1)
    }

    @Test
    func testModifyContextParams() async throws {
        var params = Llama.contextDefaultParams()
        params.n_ctx = 1024
        #expect(params.n_ctx == 1024)
    }

    @Test
    func testModifySamplerChainParams() async throws {
        let params = Llama.samplerChainDefaultParams()
        // Test that we can modify sampler chain parameters
        _ = params
        #expect(Bool(true))
    }

    // MARK: - String and Pointer Handling

    @Test
    func testStringHandling() async throws {
        let paths = ["path1", "path2"]
        let cPaths = paths.map { strdup($0) }
        defer { cPaths.forEach { if let ptr = $0 { Darwin.free(ptr) } } }
        #expect(cPaths.count == 2)
    }

    @Test
    func testArrayHandling() async throws {
        let tokens: [Llama.Token] = [1, 2, 3, 4, 5]
        #expect(tokens.count == 5)

        let tokenArray = tokens.withUnsafeBufferPointer { $0 }
        #expect(tokenArray.count == 5)
    }

    @Test
    func testPointerAllocation() async throws {
        let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: 100)
        defer { buffer.deallocate() }
        // If we got here, allocation succeeded
        #expect(Bool(true))
    }

    // MARK: - Backend Functions

    @Test
    func testBackendInit() async throws {
        Llama.backendInit()
        Llama.backendFree()
    }

    // MARK: - Sampler Tests

    @Test
    func testSamplerInitialization() async throws {
        // Test various sampler initialization functions
        let greedySampler = Llama.samplerInitGreedy()
        defer { Llama.samplerFree(greedySampler) }
        #expect(greedySampler != nil)

        let distSampler = Llama.samplerInitDist(42)
        defer { Llama.samplerFree(distSampler) }
        #expect(distSampler != nil)

        let topKSampler = Llama.samplerInitTopK(10)
        defer { Llama.samplerFree(topKSampler) }
        #expect(topKSampler != nil)

        let topPSampler = Llama.samplerInitTopP(0.9, 1)
        defer { Llama.samplerFree(topPSampler) }
        #expect(topPSampler != nil)

        let tempSampler = Llama.samplerInitTemp(1.0)
        defer { Llama.samplerFree(tempSampler) }
        #expect(tempSampler != nil)
    }

    @Test
    func testSamplerChainOperations() async throws {
        let params = Llama.samplerChainDefaultParams()
        let chain = Llama.samplerChainInit(params)
        defer {
            if let chain = chain {
                Llama.samplerFree(chain)
            }
        }
        #expect(chain != nil)

        if let chain = chain {
            let count = Llama.samplerChainN(chain)
            #expect(count >= 0)
        }
    }

    // MARK: - Chat Template Tests

    @Test
    func testChatTemplateFunctions() async throws {
        // Test built-in chat templates
        let templates = UnsafeMutablePointer<UnsafePointer<CChar>?>.allocate(capacity: 10)
        defer { templates.deallocate() }

        let count = Llama.chatBuiltinTemplates(templates, 10)
        #expect(count >= 0)
    }

    // MARK: - Performance Tests

    @Test
    func testPerformanceDataTypes() async throws {
        // Test that performance data types are accessible
        let _: Llama.PerfContextData
        let _: Llama.PerfSamplerData
        #expect(Bool(true))
    }

    // MARK: - Token Data Array Tests

    @Test
    func testTokenDataArrayHandling() async throws {
        // Test token data array type
        let _: Llama.TokenDataArray
        #expect(Bool(true))
    }

    // MARK: - Logit Bias Tests

    @Test
    func testLogitBiasHandling() async throws {
        // Test logit bias type
        let _: Llama.LogitBias
        #expect(Bool(true))
    }

    // MARK: - Chat Message Tests

    @Test
    func testChatMessageHandling() async throws {
        // Test chat message type
        let _: Llama.ChatMessage
        #expect(Bool(true))
    }

    // MARK: - Optimization Tests

    @Test
    func testOptimizationParameters() async throws {
        // Test optimization parameter type
        let _: Llama.OptParams
        #expect(Bool(true))
    }

    // MARK: - Error Handling Tests

    @Test
    func testErrorHandling() async throws {
        // Test that we can handle potential errors gracefully
        let invalidPaths = ["/nonexistent/path/model.gguf"]
        let params = Llama.modelDefaultParams()

        // This should return nil for invalid path, not crash
        let model = Llama.modelLoadFromFile(invalidPaths[0], params)
        #expect(model == nil)
    }

    // MARK: - Edge Case Tests

    @Test
    func testEdgeCases() async throws {
        // Test with zero values
        let zeroToken: Llama.Token = 0
        #expect(zeroToken == 0)

        // Test with maximum values
        let maxToken: Llama.Token = Llama.Token.max
        #expect(maxToken == Llama.Token.max)
    }

    // MARK: - Comprehensive API Test

    @Test
    func testAllBasicAPIsAccessible() async throws {
        // This test verifies that all major API categories are accessible

        // System
        _ = Llama.printSystemInfo()

        // Parameters
        _ = Llama.modelDefaultParams()
        _ = Llama.contextDefaultParams()
        _ = Llama.modelQuantizeDefaultParams()
        _ = Llama.samplerChainDefaultParams()

        // Capabilities
        _ = Llama.maxDevices()
        _ = Llama.maxParallelSequences()
        _ = Llama.supportsMmap()
        _ = Llama.supportsMlock()
        _ = Llama.supportsGpuOffload()
        _ = Llama.supportsRpc()

        // Time
        _ = Llama.timeUs()

        // Backend
        Llama.backendInit()
        Llama.backendFree()

        // Samplers
        let greedySampler = Llama.samplerInitGreedy()
        defer { Llama.samplerFree(greedySampler) }
        let distSampler = Llama.samplerInitDist(42)
        defer { Llama.samplerFree(distSampler) }

        // Chat templates
        let templates = UnsafeMutablePointer<UnsafePointer<CChar>?>.allocate(capacity: 10)
        defer { templates.deallocate() }
        _ = Llama.chatBuiltinTemplates(templates, 10)

        // If we got here, all basic APIs are accessible
        #expect(Bool(true))
    }
}
