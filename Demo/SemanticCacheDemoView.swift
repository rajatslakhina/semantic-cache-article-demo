import SwiftUI
import SemanticCacheKit

/// Interactive demo of `SemanticCacheKit`.
///
/// Type a prompt and "ask" — the view first consults the semantic cache; on
/// a miss it calls a small simulated generator (canned answers + artificial
/// latency, standing in for a real LLM call) and stores the result. Reworded
/// prompts that share vocabulary with a cached one come back instantly as
/// hits, with the similarity score shown.
struct SemanticCacheDemoView: View {

    private struct AskRecord: Identifiable {
        let id = UUID()
        let prompt: String
        let response: String
        let outcome: String   // "HIT (0.93)" or "MISS → generated"
        let wasHit: Bool
        let elapsedMs: Int
    }

    // Threshold note, learned the honest way: the first Simulator run of this
    // demo shipped with 0.62 here, and "Cancel an order I placed" scored
    // 0.548 against "How do I cancel my order?" — a miss on the exact pair
    // the UI suggests. With a bag-of-words hashing embedder, rephrasings
    // share less vocabulary than you'd guess; 0.50 is calibrated to this
    // embedder, and a real semantic model would want its own calibration
    // pass, not a copied constant.
    @State private var cache = SemanticCache(
        embedder: HashingEmbedder(dimension: 128),
        capacity: 32,
        similarityThreshold: 0.50
    )

    @State private var prompt: String = ""
    @State private var history: [AskRecord] = []
    @State private var metrics = CacheMetrics()
    @State private var isGenerating = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                metricsHeader
                List {
                    Section("Try these") {
                        suggestionButton("How do I cancel my order?")
                        suggestionButton("Cancel an order I placed")
                        suggestionButton("What is your refund policy?")
                    }
                    Section("Asks (newest first)") {
                        if history.isEmpty {
                            Text("No prompts asked yet.")
                                .foregroundStyle(.secondary)
                        }
                        ForEach(history) { record in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(record.prompt).font(.headline)
                                Text(record.response)
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                HStack {
                                    Text(record.outcome)
                                        .font(.caption.bold())
                                        .foregroundStyle(record.wasHit ? .green : .orange)
                                    Spacer()
                                    Text("\(record.elapsedMs) ms")
                                        .font(.caption.monospacedDigit())
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .padding(.vertical, 2)
                        }
                    }
                }
                inputBar
            }
            .navigationTitle("Semantic Cache")
        }
    }

    private var metricsHeader: some View {
        HStack(spacing: 16) {
            metricPill("Hits", "\(metrics.hits)", .green)
            metricPill("Misses", "\(metrics.misses)", .orange)
            metricPill("Hit rate", String(format: "%.0f%%", metrics.hitRate * 100), .blue)
            metricPill("Tokens saved", "\(metrics.estimatedTokensSaved)", .purple)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    private func metricPill(_ label: String, _ value: String, _ tint: Color) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.headline.monospacedDigit()).foregroundStyle(tint)
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private func suggestionButton(_ text: String) -> some View {
        Button(text) {
            prompt = text
            Task { await ask() }
        }
        .disabled(isGenerating)
    }

    private var inputBar: some View {
        HStack {
            TextField("Ask something…", text: $prompt)
                .textFieldStyle(.roundedBorder)
                .onSubmit { Task { await ask() } }
            Button {
                Task { await ask() }
            } label: {
                if isGenerating {
                    ProgressView()
                } else {
                    Image(systemName: "paperplane.fill")
                }
            }
            .disabled(prompt.trimmingCharacters(in: .whitespaces).isEmpty || isGenerating)
        }
        .padding()
    }

    private func ask() async {
        let text = prompt.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return }
        isGenerating = true
        defer { isGenerating = false }

        let start = ContinuousClock.now

        switch await cache.lookup(text) {
        case .hit(let cached, let similarity):
            let elapsed = start.duration(to: .now)
            record(
                prompt: text,
                response: cached.response,
                outcome: String(format: "HIT (similarity %.2f)", similarity),
                wasHit: true,
                elapsed: elapsed
            )
        case .miss:
            let generated = await SimulatedGenerator.respond(to: text)
            await cache.store(CachedResponse(
                prompt: text,
                response: generated.text,
                estimatedTokens: generated.estimatedTokens
            ))
            let elapsed = start.duration(to: .now)
            record(
                prompt: text,
                response: generated.text,
                outcome: "MISS → generated & cached",
                wasHit: false,
                elapsed: elapsed
            )
        }

        metrics = await cache.metricsSnapshot()
        prompt = ""
    }

    private func record(
        prompt: String,
        response: String,
        outcome: String,
        wasHit: Bool,
        elapsed: Duration
    ) {
        let ms = Int(Double(elapsed.components.attoseconds) / 1e15
            + Double(elapsed.components.seconds) * 1000)
        history.insert(
            AskRecord(prompt: prompt, response: response, outcome: outcome,
                      wasHit: wasHit, elapsedMs: ms),
            at: 0
        )
    }
}

/// A stand-in for a real LLM call: canned answers with artificial latency,
/// so the hit-vs-miss cost difference is visible in the demo.
enum SimulatedGenerator {
    struct Generated {
        let text: String
        let estimatedTokens: Int
    }

    static func respond(to prompt: String) async -> Generated {
        // Simulate model latency so misses visibly cost time.
        try? await Task.sleep(for: .milliseconds(900))

        let lower = prompt.lowercased()
        let text: String
        if lower.contains("cancel") {
            text = "Open the Orders tab, select the order, then tap Cancel Order. Orders already shipped can be returned instead."
        } else if lower.contains("refund") {
            text = "Refunds are issued to the original payment method within 5–7 business days of the return being received."
        } else if lower.contains("weather") {
            text = "I can't check live weather here, but this is where a real tool call would go."
        } else {
            text = "Here's a generated answer for: \(prompt)"
        }
        // Crude token estimate: ~¾ of a token per character / 4 chars per token.
        return Generated(text: text, estimatedTokens: max(1, text.count / 4))
    }
}

#Preview {
    SemanticCacheDemoView()
}
