// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "llama.swift",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .tvOS(.v16),
        .watchOS(.v9),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "LlamaSwift",
            targets: ["LlamaSwift"]
        )
    ],
    targets: [
        .binaryTarget(
            name: "llama-cpp",
            url:
                "https://github.com/ggml-org/llama.cpp/releases/download/b9144/llama-b9144-xcframework.zip",
            checksum: "870ea1a32d83f0debab5ba4dbf23a4fb612070b672c9dff0e117456e22b11854"
        ),
        .target(
            name: "LlamaSwift",
            dependencies: ["llama-cpp"],
            path: "Sources/LlamaSwift"
        ),
        .testTarget(
            name: "LlamaTests",
            dependencies: ["LlamaSwift"]
        ),
    ]
)
