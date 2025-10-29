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
                "https://github.com/ggml-org/llama.cpp/releases/download/b6877/llama-b6877-xcframework.zip",
            checksum: "73cebf22f2b54b51342c49d8e7c090dd3e5b594869d7c1b781cf7167823730bb"
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
