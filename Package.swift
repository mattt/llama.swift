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
            // llama.cpp v1.0.0 (b12359)
            url:
                "https://github.com/ggml-org/llama.cpp/releases/download/b12359/llama-b12359-xcframework.zip",
            checksum: "25eebedbbdef004c2da2bf4fd8daa77324d6db91bbff24d19d9da0942cbfd1b9"
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
