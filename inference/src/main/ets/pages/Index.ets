import { hilog } from '@kit.PerformanceAnalysisKit';
import { createContext, InferenceContext, ModelConfig, InputTensor, OutputTensor } from 'libinference.so';

const LOG_DOMAIN = 0x0001;
const LOG_TAG = 'UI::Inference';

@Entry
@Component
struct Index {
  @State private jsonOutput: string = '{}';
  private inferenceContext?: InferenceContext = undefined;

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
    const modelBuffer = new ArrayBuffer(1024 * 1024); // 1MB buffer
    const imageBuffer = new Float32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]); // 5 RGB pixels

    hilog.info(LOG_DOMAIN, LOG_TAG, 'Inference initiated...');

    try {
      if (!this.inferenceContext) {
        const modelConfig: ModelConfig = {
          modelData: modelBuffer,
          device: 'MOCK'
        };

        hilog.info(LOG_DOMAIN, LOG_TAG, 'Creating native context...');
        this.inferenceContext = await createContext(modelConfig);
        hilog.info(LOG_DOMAIN, LOG_TAG, 'Context created.');
      }


      const inputTensor: InputTensor = {
        data: imageBuffer,
        shape: [1, 1, 5, 3] // samples, height, width, channels = 15 elements
      };

      // run inference
      hilog.info(LOG_DOMAIN, LOG_TAG, 'Calling native runInference...');
      const outputTensor: OutputTensor = await this.inferenceContext.run(inputTensor);
      hilog.info(LOG_DOMAIN, LOG_TAG, 'Inference complete.');

      this.jsonOutput = JSON.stringify({ shape: outputTensor.shape, slice: outputTensor.data.slice(0, 10) }, null, 2);

    } catch (err) {
      const msg = `Inference failed: ${err?.message ?? String(err)}`;
      hilog.error(LOG_DOMAIN, LOG_TAG, msg);
      this.jsonOutput = JSON.stringify({ error: msg }, null, 2);
    }
  }
}
