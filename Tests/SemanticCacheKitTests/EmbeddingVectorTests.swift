import XCTest
@testable import SemanticCacheKit

final class EmbeddingVectorTests: XCTestCase {

    // MARK: - Cosine similarity, well-defined cases

    func testIdenticalVectorsHaveSimilarityOne() {
        let v = EmbeddingVector([1, 2, 3])
        let similarity = v.cosineSimilarity(to: v)
        XCTAssertNotNil(similarity)
        XCTAssertEqual(similarity ?? -99, 1.0, accuracy: 1e-9)
    }

    func testParallelVectorsOfDifferentMagnitudeHaveSimilarityOne() {
        let a = EmbeddingVector([1, 2, 3])
        let b = EmbeddingVector([2, 4, 6])
        XCTAssertEqual(a.cosineSimilarity(to: b) ?? -99, 1.0, accuracy: 1e-9)
    }

    func testOrthogonalVectorsHaveSimilarityZero() {
        let a = EmbeddingVector([1, 0])
        let b = EmbeddingVector([0, 1])
        XCTAssertEqual(a.cosineSimilarity(to: b) ?? -99, 0.0, accuracy: 1e-9)
    }

    func testOppositeVectorsHaveSimilarityMinusOne() {
        let a = EmbeddingVector([1, 1])
        let b = EmbeddingVector([-1, -1])
        XCTAssertEqual(a.cosineSimilarity(to: b) ?? -99, -1.0, accuracy: 1e-9)
    }

    func testSimilarityIsSymmetric() {
        let a = EmbeddingVector([0.3, 0.9, 0.1, 2.5])
        let b = EmbeddingVector([1.1, 0.2, 0.7, 0.4])
        let ab = a.cosineSimilarity(to: b)
        let ba = b.cosineSimilarity(to: a)
        XCTAssertNotNil(ab)
        XCTAssertEqual(ab ?? -99, ba ?? 99, accuracy: 1e-12)
    }

    // MARK: - Undefined comparisons must be nil, not 0

    func testDimensionMismatchReturnsNil() {
        let a = EmbeddingVector([1, 2, 3])
        let b = EmbeddingVector([1, 2])
        XCTAssertNil(a.cosineSimilarity(to: b))
    }

    func testEmptyVectorsReturnNil() {
        let a = EmbeddingVector([])
        let b = EmbeddingVector([])
        XCTAssertNil(a.cosineSimilarity(to: b))
    }

    func testZeroMagnitudeVectorReturnsNil() {
        let zero = EmbeddingVector([0, 0, 0])
        let real = EmbeddingVector([1, 2, 3])
        XCTAssertNil(zero.cosineSimilarity(to: real))
        XCTAssertNil(real.cosineSimilarity(to: zero))
    }

    // MARK: - Magnitude

    func testMagnitudeOfKnownVector() {
        // 3-4-5 triangle.
        XCTAssertEqual(EmbeddingVector([3, 4]).magnitude, 5.0, accuracy: 1e-12)
    }

    func testMagnitudeOfEmptyVectorIsZero() {
        XCTAssertEqual(EmbeddingVector([]).magnitude, 0.0)
    }
}
