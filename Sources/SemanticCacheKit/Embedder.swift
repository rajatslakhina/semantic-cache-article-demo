/// Anything that can turn text into an `EmbeddingVector`.
///
/// In production this seam is where a real model plugs in: Apple's
/// `NLContextualEmbedding` on-device, an MLX-hosted encoder, or a remote
/// embeddings API. The cache never knows or cares which — that provider
/// decision stays swappable, which is the whole point of putting a protocol
/// here instead of hard-coding a backend.
public protocol Embedder: Sendable {
    /// The dimensionality every vector produced by this embedder will have.
    var dimension: Int { get }

    /// Embeds `text` into a vector of exactly `dimension` values.
    func embed(_ text: String) -> EmbeddingVector
}

/// A deterministic bag-of-words hashing embedder (FNV-1a → bucket).
///
/// This is deliberately *not* a semantic model — it is a reproducible
/// stand-in that makes the cache's behavior testable on any platform,
/// including Linux CI, with zero model downloads. Two texts that share
/// vocabulary land near each other; two texts that don't, don't. Swap in a
/// real `Embedder` for production semantics.
public struct HashingEmbedder: Embedder {
    public let dimension: Int

    /// - Parameter dimension: bucket count; clamped to at least 1.
    public init(dimension: Int = 64) {
        self.dimension = Swift.max(1, dimension)
    }

    public func embed(_ text: String) -> EmbeddingVector {
        var buckets = [Double](repeating: 0, count: dimension)

        let tokens = text.lowercased().split { character in
            !(character.isLetter || character.isNumber)
        }

        for token in tokens {
            // FNV-1a over the token's UTF-8 bytes.
            var hash: UInt64 = 0xcbf2_9ce4_8422_2325
            for byte in token.utf8 {
                hash = (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01b3
            }
            let index = Int(hash % UInt64(dimension))
            buckets[index] += 1
        }

        return EmbeddingVector(buckets)
    }
}
