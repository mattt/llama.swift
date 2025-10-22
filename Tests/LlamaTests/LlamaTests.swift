import Foundation
import Testing

import Llama

@Suite("Llama Tests", .serialized)
struct LlamaTests {
    @Test
    func systemInfo() async throws {
        let systemInfo = llama_print_system_info()
        #expect(systemInfo != nil)
    }
}
