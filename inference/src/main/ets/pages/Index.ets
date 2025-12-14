import { hilog } from '@kit.PerformanceAnalysisKit';
import { runInference } from 'libinference.so';

const DOMAIN: number = 0x0666;
const LOG_TAG: string = 'NAPI::Inference';

@Entry
@Component
struct Index {
  @State output: string = '...';

  async onRunInference() {
    this.output = '...';

    try {
      const input = [1, 2, 3];
      const result = await runInference(input);

      this.output = JSON.stringify(result, null, 2);
      hilog.info(DOMAIN, LOG_TAG,  '%{public}s', JSON.stringify(result));

    } catch (err) {
      this.output = JSON.stringify(err);
      hilog.error(DOMAIN, LOG_TAG, 'Inference error: %{public}s', this.output);
    }
  }

  build() {
    Column({ space: 12 }) {
      Image($rawfile('sample.jpg'))
        .width('100%')
        .objectFit(ImageFit.Cover)
        .borderWidth(2)
        .borderRadius(4)
        .borderColor(Color.Black)

      Button('Run inference')
        .onClick(() => {
          this.onRunInference();
        })

      Text('Output')
        .fontWeight(FontWeight.Bold)

      Text(this.output)
        .fontSize(12)
        .fontFamily('monospace')
        .backgroundColor(Color.White)
        .fontColor(Color.Black)
        .borderWidth(2)
        .borderRadius(4)
        .padding(8)
        .width('100%')
    }
    .padding(16)
    .width('100%')
    .height('100%')
  }
}
