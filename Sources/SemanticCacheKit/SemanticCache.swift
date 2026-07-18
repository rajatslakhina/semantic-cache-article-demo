/// A cached prompt/response pair with a token estimate for savings accounting.
public struct CachedResponse: Sendable, Equatable {
    public let prompt: String
    public let response: String
    public let estimatedTokens: Int

    public init(prompt: String, response: String, estimatedTokens: Int) {
        self.prompt = prompt
        self.response = response
        // Negative token estimates are nonsense; clamp rather than trust.
        self.estimatedTokens = Swift.max(0, estimatedTokens)
    }
}

/// The outcome of a semantic lookup.
public enum CacheLookup: Sendable, Equatable {
    /// Best entry at or above the similarity threshold.
    case hit(CachedResponse, similarity: Double)
    /// Nothing comparable enough — the caller pays for a real generation.
    case miss
}

/// Running counters for the cache. `estimatedTokensSaved` is the sum of
/// `estimatedTokens` across every hit — the number a lead actually reports
/// when someone asks what the cache is worth.
public struct CacheMetrics: Sendable, Equatable {
    public internal(set) var hits: Int = 0
    public internal(set) var misses: Int = 0
    public internal(set) var evictions: Int = 0
    public internal(set) var estimatedTokensSaved: Int = 0

    public var lookups: Int { hits + misses }

    public var hitRate: Double {
        lookups == 0 ? 0 : Double(hits) / Double(lookups)
    }

    public init() {}
}

/// An on-device semantic cache: lookups match by embedding similarity, not
/// string equality, so "How do I cancel my order?" can be served by the
/// answer already generated for "cancel an order I placed."
///
/// Concurrency: this is an `actor`, so lookups, stores, and metrics reads
/// are serialized — two simultaneous lookups can't race the LRU bookkeeping.
///
/// Scale note, stated honestly: `lookup` is a linear scan over at most
/// `capacity` entries. For an on-device cache bounded in the hundreds,
/// that's the right trade — an ANN index (HNSW etc.) buys sublinear search
/// at the cost of build time, memory, and a dependency, and only pays for
/// itself at corpus sizes an on-device response cache should never reach.
/// Say that sentence in a system design interview and you're most of the
/// way to the follow-up question.
public actor SemanticCache {
    private struct Entry {
        let embedding: EmbeddingVector
        let cached: CachedResponse
        var lastAccessed: UInt64
    }

    /// Maximum number of entries. Clamped to `>= 0`; a capacity of 0
    /// disables storage entirely (every lookup is a miss, every store is a
    /// no-op) — useful as a kill switch without changing call sites.
    public let capacity: Int

    /// Minimum cosine similarity for a hit. Clamped to `[0, 1]`.
    public let similarityThreshold: Double

    private let embedder: any Embedder
    private var entries: [Entry] = []
    private var clock: UInt64 = 0
    private var metrics = CacheMetrics()

    public init(
        embedder: any Embedder,
        capacity: Int = 128,
        similarityThreshold: Double = 0.9
    ) {
        self.embedder = embedder
        self.capacity = Swift.max(0, capacity)
        self.similarityThreshold = Swift.min(1.0, Swift.max(0.0, similarityThreshold))
    }

    /// Looks up the best semantically-similar entry for `prompt`.
    ///
    /// A hit refreshes the entry's LRU position and credits its token
    /// estimate to `estimatedTokensSaved`. Prompts that embed to a
    /// zero-magnitude vector (e.g. the empty string) can never hit — their
    /// similarity is undefined, and undefined must not masquerade as 1.0.
    public func lookup(_ prompt: String) -> CacheLookup {
        let probe = embedder.embed(prompt)

        var bestIndex: Int?
        var bestSimilarity = -Double.infinity

        for index in entries.indices {
            guard let similarity = probe.cosineSimilarity(to: entries[index].embedding) else {
                continue
            }
            if similarity > bestSimilarity {
                bestSimilarity = similarity
                bestIndex = index
            }
        }

        guard let index = bestIndex, bestSimilarity >= similarityThreshold else {
            metrics.misses += 1
            return .miss
        }

        clock += 1
        entries[index].lastAccessed = clock
        metrics.hits += 1
        metrics.estimatedTokensSaved += entries[index].cached.estimatedTokens
        return .hit(entries[index].cached, similarity: bestSimilarity)
    }

    /// Stores a freshly generated response, evicting the least-recently-used
    /// entry if the cache is full.
    public func store(_ response: CachedResponse) {
        guard capacity > 0 else { return }

        clock += 1

        // Replace an existing entry for the same prompt text instead of
        // duplicating it.
        if let existing = entries.firstIndex(where: { $0.cached.prompt == response.prompt }) {
            entries[existing] = Entry(
                embedding: embedder.embed(response.prompt),
                cached: response,
                lastAccessed: clock
            )
            return
        }

        if entries.count >= capacity {
            // Evict least-recently-used. `min(by:)` is safe here: this branch
            // only runs when `entries` is non-empty (capacity > 0 and
            // count >= capacity), but guard anyway rather than force-unwrap.
            if let lruIndex = entries.indices.min(by: {
                entries[$0].lastAccessed < entries[$1].lastAccessed
            }) {
                entries.remove(at: lruIndex)
                metrics.evictions += 1
            }
        }

        entries.append(Entry(
            embedding: embedder.embed(response.prompt),
            cached: response,
            lastAccessed: clock
        ))
    }

    /// Current entry count.
    public var count: Int { entries.count }

    /// All cached responses, most-recently-used first (for display/debugging).
    public func snapshot() -> [CachedResponse] {
        entries
            .sorted { $0.lastAccessed > $1.lastAccessed }
            .map(\.cached)
    }

    /// A copy of the running metrics.
    public func metricsSnapshot() -> CacheMetrics { metrics }

    /// Removes all entries. Metrics are preserved — wiping the counters is a
    /// separate, deliberate act.
    public func removeAll() {
        entries.removeAll()
    }
}
