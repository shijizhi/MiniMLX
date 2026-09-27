import CoreML
import Foundation

func readFloats(_ path: String) throws -> [Float] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    precondition(data.count % MemoryLayout<Float>.size == 0)
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

func array(_ values: [Float], shape: [NSNumber]) -> MLMultiArray {
    let result = try! MLMultiArray(shape: shape, dataType: .float32)
    values.withUnsafeBufferPointer { src in
        result.dataPointer.copyMemory(from: src.baseAddress!, byteCount: values.count * MemoryLayout<Float>.size)
    }
    return result
}

let args = CommandLine.arguments
let modelPath = args.count > 1 ? args[1] : "/tmp/minimlx-gdn-recurrence-8.mlmodelc"
let inputDir = args.count > 2 ? args[2] : "/tmp/minimlx-gdn-recurrence-8-input"
let tokens = Int(args.count > 3 ? args[3] : "8")!
let heads = Int(args.count > 4 ? args[4] : "2")!
let keyDim = Int(args.count > 5 ? args[5] : "8")!
let valueDim = Int(args.count > 6 ? args[6] : "4")!

let model = try MLModel(contentsOf: URL(fileURLWithPath: modelPath))
let q = array(try readFloats("\(inputDir)/q.bin"), shape: [NSNumber(value: tokens), NSNumber(value: heads), NSNumber(value: keyDim)])
let k = array(try readFloats("\(inputDir)/k.bin"), shape: [NSNumber(value: tokens), NSNumber(value: heads), NSNumber(value: keyDim)])
let v = array(try readFloats("\(inputDir)/v.bin"), shape: [NSNumber(value: tokens), NSNumber(value: heads), NSNumber(value: valueDim)])
let g = array(try readFloats("\(inputDir)/g.bin"), shape: [NSNumber(value: tokens), NSNumber(value: heads)])
let beta = array(try readFloats("\(inputDir)/beta.bin"), shape: [NSNumber(value: tokens), NSNumber(value: heads)])
let state = array(try readFloats("\(inputDir)/state.bin"), shape: [1, NSNumber(value: heads), NSNumber(value: valueDim), NSNumber(value: keyDim)])

let provider = try MLDictionaryFeatureProvider(dictionary: [
    "q": MLFeatureValue(multiArray: q),
    "k": MLFeatureValue(multiArray: k),
    "v": MLFeatureValue(multiArray: v),
    "g": MLFeatureValue(multiArray: g),
    "beta": MLFeatureValue(multiArray: beta),
    "state_workaround": MLFeatureValue(multiArray: state),
])
_ = try model.prediction(from: provider)
let start = DispatchTime.now().uptimeNanoseconds
let result = try model.prediction(from: provider)
let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0

let names = model.modelDescription.outputDescriptionsByName.keys.sorted()
print("outputs=\(names)")
print("elapsed_ms=\(String(format: "%.3f", elapsed))")
for name in names {
    if let value = result.featureValue(for: name)?.multiArrayValue {
        let count = value.count
        let floats = value.dataPointer.bindMemory(to: Float.self, capacity: count)
        let prefix = (0..<min(8, count)).map { String(floats[$0]) }.joined(separator: ",")
        print("\(name) shape=\(value.shape) prefix=[\(prefix)]")
        try Data(bytes: floats, count: count * MemoryLayout<Float>.size).write(to: URL(fileURLWithPath: "\(inputDir)/\(name).bin"))
    }
}
