// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "embla-metal",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "EmblaMetal", targets: ["EmblaMetal"]),
        .library(name: "EmblaBenchKit", targets: ["EmblaBenchKit"]),
        .library(name: "EmblaChat", targets: ["EmblaChat"]),
        .executable(name: "embla", targets: ["embla"]),
        .executable(name: "embla-bench", targets: ["embla-bench"]),
    ],
    targets: [
        .target(name: "EmblaMetal", resources: [.copy("Kernels")]),
        .target(name: "EmblaBenchKit", dependencies: ["EmblaMetal"]),
        .target(name: "EmblaChat", dependencies: ["EmblaMetal"], resources: [.copy("Web")]),
        .executableTarget(name: "embla", dependencies: ["EmblaChat"]),
        .executableTarget(name: "embla-bench", dependencies: ["EmblaBenchKit"]),
        .testTarget(name: "EmblaMetalTests", dependencies: ["EmblaMetal"]),
        .testTarget(name: "EmblaChatTests", dependencies: ["EmblaChat"]),
    ]
)
