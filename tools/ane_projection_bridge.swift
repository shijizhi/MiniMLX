import CoreML
import Foundation

final class MiniMLXANEHandle {
    let model: MLModel
    let tokens: Int
    let inputDim: Int
    let outputDim: Int
    let inputArray: MLMultiArray
    let provider: MLDictionaryFeatureProvider

    init(path: String, tokens: Int, inputDim: Int, outputDim: Int) throws {
        let configuration = MLModelConfiguration()
        switch ProcessInfo.processInfo.environment["MINIMLX_ANE_COMPUTE_UNITS"] {
        case "ane":
            configuration.computeUnits = .cpuAndNeuralEngine
        case "gpu":
            configuration.computeUnits = .cpuAndGPU
        case "cpu":
            configuration.computeUnits = .cpuOnly
        default:
            configuration.computeUnits = .all
        }
        model = try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: configuration)
        self.tokens = tokens
        self.inputDim = inputDim
        self.outputDim = outputDim
        inputArray = try MLMultiArray(
            shape: [NSNumber(value: tokens), NSNumber(value: inputDim)],
            dataType: .float16
        )
        provider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: inputArray)
        ])
    }
}

@_cdecl("minimlx_ane_create")
public func minimlx_ane_create(
    _ modelPath: UnsafePointer<CChar>,
    _ tokens: Int32,
    _ inputDim: Int32,
    _ outputDim: Int32
) -> UnsafeMutableRawPointer? {
    do {
        let handle = try MiniMLXANEHandle(
            path: String(cString: modelPath),
            tokens: Int(tokens),
            inputDim: Int(inputDim),
            outputDim: Int(outputDim)
        )
        return Unmanaged.passRetained(handle).toOpaque()
    } catch {
        fputs("MiniMLX ANE create failed: \(error)\n", stderr)
        return nil
    }
}

@_cdecl("minimlx_ane_predict")
public func minimlx_ane_predict(
    _ opaque: UnsafeMutableRawPointer,
    _ input: UnsafeRawPointer,
    _ output: UnsafeMutableRawPointer
) -> Int32 {
    let handle = Unmanaged<MiniMLXANEHandle>.fromOpaque(opaque).takeUnretainedValue()
    let inputCount = handle.tokens * handle.inputDim
    let outputCount = handle.tokens * handle.outputDim
    do {
        handle.inputArray.dataPointer.copyMemory(from: input, byteCount: inputCount * MemoryLayout<UInt16>.size)
        let result = try handle.model.prediction(from: handle.provider)
        guard let outputArray = result.featureValue(for: "y")?.multiArrayValue else {
            return -2
        }
        output.copyMemory(from: outputArray.dataPointer, byteCount: outputCount * MemoryLayout<UInt16>.size)
        return 0
    } catch {
        fputs("MiniMLX ANE predict failed: \(error)\n", stderr)
        return -1
    }
}

@_cdecl("minimlx_ane_destroy")
public func minimlx_ane_destroy(_ opaque: UnsafeMutableRawPointer) {
    Unmanaged<MiniMLXANEHandle>.fromOpaque(opaque).release()
}
