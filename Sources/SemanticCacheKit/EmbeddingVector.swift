/// A fixed-dimension embedding vector with a bounds-safe cosine-similarity
/// implementation.
///
/// Design note: `cosineSimilarity(to:)` returns an *optional*. Dimension
/// mismatches and zero-magnitude vectors are not "similarity 0" — they are
/// "this comparison is meaningless," and collapsing the two is exactly the
/// kind of silent bug a semantic cache can't afford (it would turn garbage
/// embeddings into confident cache misses, or worse, confident hits at 0.0
/// threshold). Returning `nil` forces the caller to decide.
public struct EmbeddingVector: Sendable, Equatable {
    public let values: [Double]

    public init(_ values: [Double]) {
        self.values = values
    }

    public var dimension: Int { values.count }

    /// Euclidean (L2) magnitude.
    public var magnitude: Double {
        var sum = 0.0
        for v in values {
            sum += v * v
        }
        return sum.squareRoot()
    }

    /// Cosine similarity in `[-1, 1]`.
    ///
    /// Returns `nil` when the vectors have different dimensions, when either
    /// vector is empty, or when either has zero magnitude (an all-zero
    /// embedding — e.g. from embedding an empty string — cannot be compared
    /// directionally with anything).
    public func cosineSimilarity(to other: EmbeddingVector) -> Double? {
        guard dimension == other.dimension, dimension > 0 else { return nil }

        var dot = 0.0
        for index in 0..<dimension {
            dot += values[index] * other.values[index]
        }

        let m1 = magnitude
        let m2 = other.magnitude
        guard m1 > 0, m2 > 0 else { return nil }

        // Clamp to [-1, 1] to absorb floating-point drift at the boundaries.
        let raw = dot / (m1 * m2)
        return Swift.min(1.0, Swift.max(-1.0, raw))
    }
}
