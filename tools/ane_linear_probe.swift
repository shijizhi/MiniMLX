import CoreML
import Foundation

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted[sorted.count / 2]
}

guard CommandLine.arguments.count >= 2 else {
    fputs("usage: ane_linear_probe <model.mlmodelc> [tokens] [inputDim]\n", stderr)
    exit(2)
}

let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
let tokens = Int(CommandLine.arguments.dropFirst(2).first ?? "2048") ?? 2048
let inputDim = Int(CommandLine.arguments.dropFirst(3).first ?? "1024") ?? 1024
let outputDim = Int(CommandLine.arguments.dropFirst(5).first ?? "2048") ?? 2048
let dumpPath = CommandLine.arguments.dropFirst(4).first.map { String($0) }

let modes: [(String, MLComputeUnits)] = [
    ("cpu", .cpuOnly),
    ("gpu", .cpuAndGPU),
    ("ane", .cpuAndNeuralEngine),
    ("all", .all),
]

let inputCount = tokens * inputDim
let input = try MLMultiArray(shape: [NSNumber(value: tokens), NSNumber(value: inputDim)], dataType: .float16)
for index in 0..<inputCount {
    input[index] = NSNumber(value: Float((index % 97) - 48) / 97.0)
}

for (name, computeUnits) in modes {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    let provider = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: input)])

    for _ in 0..<3 {
        _ = try model.prediction(from: provider)
    }

    var samples: [Double] = []
    var lastOutput: MLMultiArray?
    for _ in 0..<10 {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try model.prediction(from: provider)
        lastOutput = result.featureValue(for: "y")?.multiArrayValue
        samples.append((CFAbsoluteTimeGetCurrent() - start) * 1000.0)
    }
    let medianText = String(format: "%.3f", median(samples))
    let samplesText = samples.map { String(format: "%.3f", $0) }.joined(separator: ",")
    print("mode=\(name) median_ms=\(medianText) samples_ms=\(samplesText)")

    if name == "ane", let dumpPath, let lastOutput {
        var data = Data()
        for index in 0..<lastOutput.count {
            var value = Float(truncating: lastOutput[index])
            data.append(Data(bytes: &value, count: MemoryLayout<Float>.size))
        }
        try data.write(to: URL(fileURLWithPath: dumpPath))
        print("dumped_output=\(dumpPath)")
    }
}
