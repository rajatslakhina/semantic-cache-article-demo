import XCTest
@testable import SemanticCacheKit

final class HashingEmbedderTests: XCTestCase {

    func testEmbeddingIsDeterministic() {
        let embedder = HashingEmbedder(dimension: 64)
        let a = embedder.embed("how do I cancel my order")
        let b = embedder.embed("how do I cancel my order")
        XCTAssertEqual(a, b)
    }

    func testEmbeddingIsCaseInsensitive() {
        let embedder = HashingEmbedder(dimension: 64)
        XCTAssertEqual(
            embedder.embed("Cancel My Order"),
            embedder.embed("cancel my order")
        )
    }

    func testPunctuationDoesNotChangeTokens() {
        let embedder = HashingEmbedder(dimension: 64)
        XCTAssertEqual(
            embedder.embed("cancel, my order!"),
            embedder.embed("cancel my order")
        )
    }

    func testEmptyStringEmbedsToZeroVector() {
        let embedder = HashingEmbedder(dimension: 8)
        let v = embedder.embed("")
        XCTAssertEqual(v.dimension, 8)
        XCTAssertEqual(v.magnitude, 0.0)
    }

    func testDimensionIsClampedToAtLeastOne() {
        XCTAssertEqual(HashingEmbedder(dimension: 0).dimension, 1)
        XCTAssertEqual(HashingEmbedder(dimension: -5).dimension, 1)
        // A degenerate 1-bucket embedder must still produce valid vectors.
        let v = HashingEmbedder(dimension: 0).embed("hello world")
        XCTAssertEqual(v.dimension, 1)
        XCTAssertEqual(v.values[0], 2.0) // two tokens, one bucket
    }

    func testProducedVectorsAlwaysMatchDeclaredDimension() {
        let embedder = HashingEmbedder(dimension: 32)
        XCTAssertEqual(embedder.embed("a").dimension, 32)
        XCTAssertEqual(embedder.embed("many more tokens here now").dimension, 32)
    }

    func testSharedVocabularyScoresHigherThanDisjointVocabulary() {
        let embedder = HashingEmbedder(dimension: 128)
        let base = embedder.embed("how do I cancel my order")
        let related = embedder.embed("cancel my order please")
        let unrelated = embedder.embed("what is the weather in Berlin")

        let relatedScore = base.cosineSimilarity(to: related) ?? -1
        let unrelatedScore = base.cosineSimilarity(to: unrelated) ?? -1
        XCTAssertGreaterThan(relatedScore, unrelatedScore)
    }
}
