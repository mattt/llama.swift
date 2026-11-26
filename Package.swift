// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "llama.swift",
    platforms: [
        .macOS("13.3"),
        .iOS("16.4"),
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
            // llama.cpp v1.1.0 (b12774)
            url:
                "https://github.com/ggml-org/llama.cpp/releases/download/b12774/llama-b12774-xcframework.zip",
            checksum: "034672ad21972aa0408930fe1d5ec5f78b7e2497a35d1df18f01aa987c54a07e"
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
