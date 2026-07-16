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
                "https://github.com/ggml-org/llama.cpp/releases/download/b10038/llama-b10038-xcframework.zip",
            checksum: "0182648b614a25780f1f95aae16e3d3da0d8551f558c512c2216f23da5de9abc"
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
