import XCTest
@testable import SemanticCacheKit

final class SemanticCacheTests: XCTestCase {

    private func makeCache(
        capacity: Int = 8,
        threshold: Double = 0.85
    ) -> SemanticCache {
        SemanticCache(
            embedder: HashingEmbedder(dimension: 128),
            capacity: capacity,
            similarityThreshold: threshold
        )
    }

    // MARK: - Basic hit/miss behavior

    func testLookupOnEmptyCacheMisses() async {
        let cache = makeCache()
        let result = await cache.lookup("anything at all")
        XCTAssertEqual(result, .miss)
    }

    func testExactPromptHitsAfterStore() async {
        let cache = makeCache()
        let stored = CachedResponse(
            prompt: "how do I cancel my order",
            response: "Open Orders, tap the order, tap Cancel.",
            estimatedTokens: 120
        )
        await cache.store(stored)

        let result = await cache.lookup("how do I cancel my order")
        guard case .hit(let cached, let similarity) = result else {
            return XCTFail("Expected a hit, got \(result)")
        }
        XCTAssertEqual(cached, stored)
        XCTAssertEqual(similarity, 1.0, accuracy: 1e-9)
    }

    func testSimilarWordingHits() async {
        let cache = makeCache(threshold: 0.6)
        await cache.store(CachedResponse(
            prompt: "how do I cancel my order",
            response: "Open Orders, tap the order, tap Cancel.",
            estimatedTokens: 120
        ))

        let result = await cache.lookup("cancel my order how do I do it")
        guard case .hit = result else {
            return XCTFail("Expected a semantic hit for reworded prompt, got \(result)")
        }
    }

    func testUnrelatedPromptMisses() async {
        let cache = makeCache(threshold: 0.85)
        await cache.store(CachedResponse(
            prompt: "how do I cancel my order",
            response: "Open Orders, tap the order, tap Cancel.",
            estimatedTokens: 120
        ))

        let result = await cache.lookup("what is the weather in Berlin today")
        XCTAssertEqual(result, .miss)
    }

    func testEmptyPromptNeverHitsAndNeverCrashes() async {
        let cache = makeCache(threshold: 0.0)
        await cache.store(CachedResponse(
            prompt: "real prompt",
            response: "real answer",
            estimatedTokens: 10
        ))
        // Empty string embeds to a zero vector; similarity is undefined (nil),
        // which must surface as a miss even at threshold 0 — not a crash and
        // not a bogus hit.
        let result = await cache.lookup("")
        XCTAssertEqual(result, .miss)
    }

    // MARK: - LRU eviction

    func testEvictsLeastRecentlyUsedWhenFull() async {
        let cache = makeCache(capacity: 2, threshold: 0.99)
        let first = CachedResponse(prompt: "alpha bravo charlie", response: "1", estimatedTokens: 1)
        let second = CachedResponse(prompt: "delta echo foxtrot", response: "2", estimatedTokens: 1)
        let third = CachedResponse(prompt: "golf hotel india", response: "3", estimatedTokens: 1)

        await cache.store(first)
        await cache.store(second)

        // Touch `first` so `second` becomes the LRU entry.
        _ = await cache.lookup("alpha bravo charlie")

        await cache.store(third)

        let count = await cache.count
        XCTAssertEqual(count, 2)

        // `second` should be gone; `first` and `third` should hit.
        let hitFirst = await cache.lookup("alpha bravo charlie")
        let missSecond = await cache.lookup("delta echo foxtrot")
        let hitThird = await cache.lookup("golf hotel india")

        guard case .hit = hitFirst else { return XCTFail("first was evicted but should not be") }
        XCTAssertEqual(missSecond, .miss, "second was the LRU entry and should be evicted")
        guard case .hit = hitThird else { return XCTFail("third was just stored and must be present") }
    }

    func testStoringSamePromptReplacesInsteadOfDuplicating() async {
        let cache = makeCache(capacity: 4, threshold: 0.99)
        await cache.store(CachedResponse(prompt: "same prompt", response: "old", estimatedTokens: 5))
        await cache.store(CachedResponse(prompt: "same prompt", response: "new", estimatedTokens: 7))

        let count = await cache.count
        XCTAssertEqual(count, 1)

        let result = await cache.lookup("same prompt")
        guard case .hit(let cached, _) = result else {
            return XCTFail("Expected a hit")
        }
        XCTAssertEqual(cached.response, "new")
    }

    // MARK: - Capacity edge cases

    func testZeroCapacityDisablesStorage() async {
        let cache = makeCache(capacity: 0)
        await cache.store(CachedResponse(prompt: "p", response: "r", estimatedTokens: 1))
        let count = await cache.count
        XCTAssertEqual(count, 0)
        let result = await cache.lookup("p")
        XCTAssertEqual(result, .miss)
    }

    func testNegativeCapacityIsClampedToZero() async {
        let cache = makeCache(capacity: -3)
        let capacity = await cache.capacity
        XCTAssertEqual(capacity, 0)
    }

    func testThresholdIsClampedToUnitInterval() async {
        let low = SemanticCache(embedder: HashingEmbedder(), similarityThreshold: -2)
        let high = SemanticCache(embedder: HashingEmbedder(), similarityThreshold: 7)
        let lowThreshold = await low.similarityThreshold
        let highThreshold = await high.similarityThreshold
        XCTAssertEqual(lowThreshold, 0.0)
        XCTAssertEqual(highThreshold, 1.0)
    }

    // MARK: - Metrics

    func testMetricsCountHitsMissesAndTokensSaved() async {
        let cache = makeCache(threshold: 0.99)
        await cache.store(CachedResponse(prompt: "alpha bravo", response: "a", estimatedTokens: 40))

        _ = await cache.lookup("alpha bravo")            // hit (+40)
        _ = await cache.lookup("alpha bravo")            // hit (+40)
        _ = await cache.lookup("totally different text") // miss

        let metrics = await cache.metricsSnapshot()
        XCTAssertEqual(metrics.hits, 2)
        XCTAssertEqual(metrics.misses, 1)
        XCTAssertEqual(metrics.estimatedTokensSaved, 80)
        XCTAssertEqual(metrics.lookups, 3)
        XCTAssertEqual(metrics.hitRate, 2.0 / 3.0, accuracy: 1e-12)
    }

    func testEvictionIsCountedInMetrics() async {
        let cache = makeCache(capacity: 1, threshold: 0.99)
        await cache.store(CachedResponse(prompt: "one two", response: "1", estimatedTokens: 1))
        await cache.store(CachedResponse(prompt: "three four", response: "2", estimatedTokens: 1))

        let metrics = await cache.metricsSnapshot()
        XCTAssertEqual(metrics.evictions, 1)
    }

    func testHitRateOnFreshCacheIsZeroNotNaN() async {
        let cache = makeCache()
        let metrics = await cache.metricsSnapshot()
        XCTAssertEqual(metrics.hitRate, 0.0)
        XCTAssertFalse(metrics.hitRate.isNaN)
    }

    // MARK: - Negative token clamp

    func testNegativeTokenEstimateIsClampedToZero() {
        let response = CachedResponse(prompt: "p", response: "r", estimatedTokens: -50)
        XCTAssertEqual(response.estimatedTokens, 0)
    }

    // MARK: - Snapshot ordering

    func testSnapshotIsMostRecentlyUsedFirst() async {
        let cache = makeCache(capacity: 3, threshold: 0.99)
        await cache.store(CachedResponse(prompt: "alpha bravo", response: "1", estimatedTokens: 1))
        await cache.store(CachedResponse(prompt: "charlie delta", response: "2", estimatedTokens: 1))
        _ = await cache.lookup("alpha bravo") // refresh alpha

        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.map(\.response), ["1", "2"])
    }

    // MARK: - Concurrency smoke test

    func testConcurrentLookupsAndStoresDoNotCorruptState() async {
        let cache = makeCache(capacity: 16, threshold: 0.99)
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<50 {
                group.addTask {
                    await cache.store(CachedResponse(
                        prompt: "prompt number \(i)",
                        response: "response \(i)",
                        estimatedTokens: 1
                    ))
                }
                group.addTask {
                    _ = await cache.lookup("prompt number \(i)")
                }
            }
        }
        let count = await cache.count
        XCTAssertLessThanOrEqual(count, 16, "capacity must never be exceeded")
        let metrics = await cache.metricsSnapshot()
        XCTAssertEqual(metrics.lookups, 50)
    }
}
