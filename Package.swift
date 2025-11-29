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
                "https://github.com/ggml-org/llama.cpp/releases/download/b7198/llama-b7198-xcframework.zip",
            checksum: "721415caa524b99b04454b7466e5c0898dd4db96b67290fc113583f927de9ed1"
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
