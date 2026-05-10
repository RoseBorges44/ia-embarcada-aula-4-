#include "output_handler.h"

#include "constants.h"
#include "tensorflow/lite/micro/micro_log.h"

static const char* Classify(float lux_score) {
  if (lux_score < kThresholdEscuro)    return "ESCURO";
  if (lux_score < kThresholdPenumbra)  return "PENUMBRA";
  if (lux_score < kThresholdClaro)     return "CLARO";
  return "MUITO_CLARO";
}

void HandleOutput(int adc_raw, float adc_norm, float lux_score) {
  MicroPrintf("ADC=%4d (norm=%.3f) | lux_score=%.3f | classe=%s",
              adc_raw, static_cast<double>(adc_norm),
              static_cast<double>(lux_score), Classify(lux_score));
}
