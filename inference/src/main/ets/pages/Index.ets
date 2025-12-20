import { common } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { createContext, InferenceContext, ModelConfig, InputTensor, OutputTensor } from 'libinference.so';
import { image } from '@kit.ImageKit';

import { imagenetClassName } from './imagenet1k_id_to_class';

const LOG_DOMAIN = 0x0001;
const LOG_TAG = 'UI::Inference';

interface Prediction {
  index: number;
  probability: number;
}

export type NormalizeMode = 'minus1to1' | 'imagenet' | 'none';


@Entry
@Component
struct Index {
  @State private jsonOutput: string = '{}';
  private context?: common.UIAbilityContext = undefined;
  private inferenceContext?: InferenceContext = undefined;
  private imageBuffer?: Float32Array = undefined;
  private modelBuffer?: ArrayBuffer = undefined;

  aboutToAppear(): void {
    this.context = this.getUIContext().getHostContext() as common.UIAbilityContext;
  }

  build() {
    Column({ space: 12 }) {
      Image($rawfile('sample.jpg'))
        .width('100%')
        .objectFit(ImageFit.Contain)
        .borderWidth(2)
        .borderRadius(5)
        .borderColor(Color.Black)

      Button('Run inference')
        .onClick(() => {
          this.onRunInference();
        })

      Text(this.jsonOutput)
        .fontSize(12)
        .fontFamily('monospace')
        .backgroundColor(Color.White)
        .fontColor(Color.Black)
        .borderWidth(2)
        .borderRadius(5)
        .padding(15)
        .width('100%')
    }
    .padding(16)
    .width('100%')
    .height('100%')
  }

  async onRunInference() {
    if (!this.modelBuffer) {
      this.modelBuffer = await this.readRawFile('mobilenetv2.ms');
    }

    const imageWidth = 224;
    const imageHeight = 224;

    if (!this.imageBuffer) {
      const imageBytes = await this.readRawFile('sample.jpg');
      const pm = await this.decode(imageBytes, imageWidth, imageHeight);
      this.imageBuffer = await this.convert(pm, imageWidth, imageHeight);
    }

    try {
      // create context
      if (!this.inferenceContext) {
        const modelConfig: ModelConfig = {
          modelData: this.modelBuffer,
          device: 'MOCK'
        };

        this.inferenceContext = await createContext(modelConfig);
      }

      // prepare input
      const inputTensor: InputTensor = {
        data: this.imageBuffer,
        shape: new Uint32Array([1, 3, imageHeight, imageWidth]) // samples, channels, height, width (NCHW)
      };

      // run inference => output
      const outputTensor: OutputTensor = await this.inferenceContext.run(inputTensor);

      // top classes
      const topClasses = this.topKSoftMax(outputTensor.data, 10)
        .map(p => `${p.index}: ${imagenetClassName(p.index)} - ${(p.probability * 100).toFixed(2)}%`)
      this.jsonOutput = JSON.stringify({ prediction: topClasses }, null, 2);

    } catch (err) {
      const msg = `Inference failed: ${err?.message ?? String(err)}`;
      hilog.error(LOG_DOMAIN, LOG_TAG, msg);
      this.jsonOutput = JSON.stringify({ error: msg }, null, 2);
    }
  }

  private topKSoftMax(logits: Float32Array, k: number = 5): Prediction[] {
    const size = logits.length;
    if (size === 0) {
      return [];
    }

    let maxLogit = -Infinity;
    for (let i = 0; i < size; i++) {
      if (logits[i] > maxLogit) {
        maxLogit = logits[i];
      }
    }

    const exps = new Float32Array(size);
    let sumExps = 0;
    for (let i = 0; i < size; i++) {
      const val = Math.exp(logits[i] - maxLogit);
      exps[i] = val;
      sumExps += val;
    }

    const probabilities: Prediction[] = new Array(size);
    for (let i = 0; i < size; i++) {
      probabilities[i] = {
        index: i,
        probability: exps[i] / sumExps
      };
    }

    return probabilities
      .sort((a, b) => b.probability - a.probability)
      .slice(0, Math.min(k, size));
  }

  private async readRawFile(fileName: string): Promise<ArrayBuffer> {
    if (!this.context) {
      return new ArrayBuffer(0);
    }

    try {
      const u8: Uint8Array = await this.context.resourceManager.getRawFileContent(fileName);
      return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
    } catch (err) {
      console.error(`Failed to load data: ${JSON.stringify(err)}}`);
      return new ArrayBuffer(0);
    }
  }

  private async decode(buf: ArrayBuffer, w: number, h: number): Promise<image.PixelMap> {
    const src: image.ImageSource = image.createImageSource(buf);
    try {
      const opts: image.DecodingOptions = {
        desiredSize: { width: w, height: h },
        desiredPixelFormat: image.PixelMapFormat.RGBA_8888
      };
      return await src.createPixelMap(opts);
    } finally {
      src.release();
    }
  }

  private async convert(pm: image.PixelMap, w: number, h: number,
    normalize: NormalizeMode = 'imagenet'): Promise<Float32Array> {
    const pixelBytes = new ArrayBuffer(w * h * 4);
    await pm.readPixelsToBuffer(pixelBytes);
    const u8 = new Uint8Array(pixelBytes);

    const out = new Float32Array(1 * 3 * h * w);
    const plane = h * w;

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // RGBA -> RGB, write NCHW
    for (let y = 0; y < h; ++y) {
      for (let x = 0; x < w; ++x) {
        const p = (y * w + x) * 4;

        const r = u8[p + 0];
        const g = u8[p + 1];
        const b = u8[p + 2];
        // skip alpha channel

        let rf = r / 255.0;
        let gf = g / 255.0;
        let bf = b / 255.0;

        if (normalize === 'minus1to1') {
          rf = (rf - 0.5) * 2.0;
          gf = (gf - 0.5) * 2.0;
          bf = (bf - 0.5) * 2.0;
        } else if (normalize === 'imagenet') {
          rf = (rf - mean[0]) / std[0];
          gf = (gf - mean[1]) / std[1];
          bf = (bf - mean[2]) / std[2];
        }

        const idx = y * w + x;
        out[0 * plane + idx] = rf;
        out[1 * plane + idx] = gf;
        out[2 * plane + idx] = bf;
      }
    }

    return out;
  }
}
