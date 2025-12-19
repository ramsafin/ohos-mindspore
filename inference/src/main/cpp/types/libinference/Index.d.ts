export type Shape = Uint32Array;

export type Device = 'CPU' | 'MOCK'; // later: 'NPU'

/** Config options required to load the inference model */
export interface ModelConfig {
  modelData: ArrayBuffer; // contains model binary data
  device: Device; // runtime device (e.g., CPU, MOCK)
}

/** Input tensor passed to native inference */
export interface InputTensor {
  data: Float32Array; // input tensor data, expected to be a flattened pixel map
  shape: Shape; // input tensor dimensions, e.g., [1, 224, 224, 3]
}

/** Output tensor returned after inference */
// why do we have different types for tensors?
export interface OutputTensor {
  data: Float32Array; // output tensor data, expected to be probabilities of classes
  shape: Shape; // output tensor dimensions, e.g. [1, 1000]
}

export interface InferenceContext {
  /**
   * Runs inference using the loaded model on the provided image tensor data.
   * This is an async operation that can throw a descriptive error
   * if the model isn't loaded, input is wrong size, or runtime error occurs..
   *
   * @param input The input data structured as an InputTensor.
   * @returns A Promise that resolves with an OutputTensor.
   * @throws {Error} An error if inference fails.
   */
  run(input: InputTensor): Promise<OutputTensor>;
}

/**
 * Create a context by loading and compiling an inference model from binary data.
 * This is an async operation that can throw a descriptive error
 * if the model is corrupt, incompatible, or the device fails to initialize.
 *
 * @param config The config containing the model data buffer and options.
 * @returns A Promise that resolves when the model is successfully loaded.
 * @throws {Error} An error if loading fails.
 */
export function createContext(config: ModelConfig): Promise<InferenceContext>;




