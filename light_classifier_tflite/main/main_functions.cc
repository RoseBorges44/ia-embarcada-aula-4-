#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_adc/adc_oneshot.h"

#include "main_functions.h"
#include "model.h"
#include "constants.h"
#include "output_handler.h"

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];

adc_oneshot_unit_handle_t adc1_handle = nullptr;
constexpr adc_unit_t    kAdcUnit    = ADC_UNIT_1;
constexpr adc_channel_t kAdcChannel = ADC_CHANNEL_0;  // GPIO1 no ESP32-S3
}  // namespace

static void SetupAdc() {
  adc_oneshot_unit_init_cfg_t init_cfg = {};
  init_cfg.unit_id = kAdcUnit;
  ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_cfg, &adc1_handle));

  adc_oneshot_chan_cfg_t chan_cfg = {};
  chan_cfg.atten    = ADC_ATTEN_DB_12;     // ate ~3.1V (cobre 0..3.3V do divisor)
  chan_cfg.bitwidth = ADC_BITWIDTH_DEFAULT; // 12 bits no S3
  ESP_ERROR_CHECK(adc_oneshot_config_channel(adc1_handle, kAdcChannel, &chan_cfg));
}

void setup() {
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d != supported %d", model->version(),
                TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    MicroPrintf("AddFullyConnected failed");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed");
    return;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  SetupAdc();
  MicroPrintf("Light classifier pronto. Lendo ADC1_CH0 (GPIO1).");
}

void loop() {
  // 1. Le o ADC raw (0..4095).
  int adc_raw = 0;
  if (adc_oneshot_read(adc1_handle, kAdcChannel, &adc_raw) != ESP_OK) {
    MicroPrintf("Falha lendo ADC");
    return;
  }
  float adc_norm = static_cast<float>(adc_raw) / kAdcMaxRaw;

  // 2. Quantiza para int8 e roda inferencia.
  int8_t in_q = adc_norm / input->params.scale + input->params.zero_point;
  input->data.int8[0] = in_q;
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed (adc_norm=%.3f)", static_cast<double>(adc_norm));
    return;
  }

  // 3. Dequantiza output e classifica.
  int8_t out_q = output->data.int8[0];
  float lux_score =
      (out_q - output->params.zero_point) * output->params.scale;

  HandleOutput(adc_raw, adc_norm, lux_score);
}
