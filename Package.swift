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
                "https://github.com/ggml-org/llama.cpp/releases/download/b7170/llama-b7170-xcframework.zip",
            checksum: "ea51e82b607536499f644d4f0be5e9879da379e5d9410d20c925c27413945cfb"
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
