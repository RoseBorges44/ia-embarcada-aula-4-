#ifndef LIGHT_CLASSIFIER_CONSTANTS_H_
#define LIGHT_CLASSIFIER_CONSTANTS_H_

// LDR ligado em divisor de tensao com 10k pulldown,
// lido pelo ADC1 canal 0 (GPIO1 no ESP32-S3 DevKitC-1).
// O canal e a unidade ficam definidos no main_functions.cc para nao
// vazar dependencias do ESP-IDF neste header.

// O ADC do ESP32-S3 e 12 bits (0..4095).
extern const float kAdcMaxRaw;

// Limiares para classificar o output [0,1] do modelo em 4 zonas.
extern const float kThresholdEscuro;
extern const float kThresholdPenumbra;
extern const float kThresholdClaro;

#endif  // LIGHT_CLASSIFIER_CONSTANTS_H_
