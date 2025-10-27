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
            name: "Llama",
            targets: ["Llama"]
        )
    ],
    targets: [
        .binaryTarget(
            name: "llama-cpp",
            url:
                "https://github.com/ggml-org/llama.cpp/releases/download/b6857/llama-b6857-xcframework.zip",
            checksum: "b0fd68307998a53e61845c48ed07de4dc8bc3927576c14187238c0a87fbbfd78"
        ),
        .target(
            name: "Llama",
            dependencies: ["llama-cpp"],
            path: "Sources/Llama"
        ),
        .testTarget(
            name: "LlamaTests",
            dependencies: ["Llama"]
        ),
    ]
)
