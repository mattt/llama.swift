import Foundation
import Testing

@testable import Llama

@Suite("Llama Tests")
struct LlamaTests {
    // MARK: - Type Tests

    @Test
    func testTypeAliasesExist() async throws {
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

        // If we got here, all basic APIs are accessible
        #expect(Bool(true))
    }
}
