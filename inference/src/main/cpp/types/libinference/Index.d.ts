export interface InferenceResult {
  ok: boolean;
  error?: string;
  output?: number[];
}

/**
 * Run inference asynchronously.
 * Returns inference results.
 */
export const runInference: (input: number[]) => Promise<InferenceResult>;