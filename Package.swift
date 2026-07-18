// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "SemanticCacheKit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        // Library only — the runnable app lives in Demo/Demo.xcodeproj,
        // never as an executable target inside this package.
        .library(name: "SemanticCacheKit", targets: ["SemanticCacheKit"])
    ],
    targets: [
        .target(name: "SemanticCacheKit"),
        .testTarget(
            name: "SemanticCacheKitTests",
            dependencies: ["SemanticCacheKit"]
        )
    ]
)
