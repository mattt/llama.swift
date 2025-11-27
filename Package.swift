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
                "https://github.com/ggml-org/llama.cpp/releases/download/b7176/llama-b7176-xcframework.zip",
            checksum: "4435debb66f774443bcc9f96b31b74486e495d7f3d27207486c174c445223d1c"
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
