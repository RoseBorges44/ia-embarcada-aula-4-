# IA Embarcada — Atividade Avaliativa Prática (4/6)

**Disciplina:** IA Embarcada e Modelos Compactos
**Aluna:** Rosemeri Borges
**Tema:** TensorFlow Lite Micro no ESP32-S3 (simulado pelo Wokwi)

Reprodução do exemplo *Hello World* do TensorFlow Lite Micro sobre o
`esp-tflite-micro` (obrigatório) e implementação de uma aplicação extra
de classificação de luminosidade com sensor LDR (extra +1 ponto).

## Aplicações

### 1. Hello World — regressor de seno

Modelo Keras (`Dense(16) → Dense(16) → Dense(1)`) treinado para
aproximar `y = sin(x)` em `x ∈ [0, 2π]`, quantizado para int8 e
embarcado no ESP32-S3 como um array C de 3.344 bytes. O firmware
gera valores de `x` em loop e imprime `(x, ŷ)` no serial.

### 2. Light Classifier — classificador de luminosidade (extra)

Reaproveita a mesma arquitetura/op-resolver do Hello World, mas:

- Lê um sensor LDR (`wokwi-photoresistor-sensor`) no ADC1_CH0 (GPIO1)
  via `esp_adc/adc_oneshot`.
- Foi retreinado com dataset sintético próprio
  (`adc_norm ~ U(0,1)` → `lux = sqrt(adc_norm) + ruído gaussiano`).
- Classifica em 4 zonas no firmware (`ESCURO/PENUMBRA/CLARO/MUITO_CLARO`)
  por limiares 0,25 / 0,50 / 0,75 sobre o `lux_score` predito.

## Estrutura do repositório

```
.
├── README.md                              ← este arquivo
├── relatorio.md / .docx / .pdf            ← relatório completo de análise (8 seções)
├── tflite_hello_world_training.ipynb      ← treino + quantização do Hello World
├── light_classifier_training.ipynb        ← treino + quantização do extra
├── md_to_docx.py                          ← conversor markdown → Word usado no relatório
├── print_hello_world.jpg                  ← print do Wokwi rodando o Hello World
├── print_light_classifier.jpg             ← print do Wokwi rodando o Light Classifier
│
├── hello_world_tflite/                    ← projeto ESP-IDF do Hello World
│   ├── CMakeLists.txt
│   ├── diagram.json                       ← circuito Wokwi (só ESP32-S3)
│   ├── wokwi.toml                         ← config do simulador
│   ├── sdkconfig.defaults                 ← target = esp32s3
│   ├── hello_world_int8.tflite            ← modelo quantizado (artefato do treino)
│   ├── tflite_to_cc.py                    ← .tflite → array C (substitui o `xxd`)
│   └── main/
│       ├── main.cc                        ← app_main → setup/loop a cada 500ms
│       ├── main_functions.cc/h            ← pipeline TFLM (parse, resolver, interpreter)
│       ├── model.cc/h                     ← g_model[] alinhado em 8 bytes
│       ├── constants.cc/h                 ← kXrange = 2π, kInferencesPerCycle = 20
│       └── output_handler.cc/h            ← MicroPrintf no serial
│
└── light_classifier_tflite/               ← projeto ESP-IDF do extra
    ├── ... (mesma estrutura)
    └── main/
        ├── main_functions.cc              ← agora lê ADC ao invés de gerar x sintético
        ├── model.cc                       ← g_model[] do regressor de luminosidade
        └── output_handler.cc              ← classifica por limiares e imprime classe
```

Os artefatos de build (`build/`, `managed_components/`, `dependencies.lock`,
`sdkconfig.old`) são ignorados pelo `.gitignore` — basta executar
`idf.py build` que tudo é regenerado.

## Como reproduzir

### Pré-requisitos

- ESP-IDF v5.4.4 (toolchain Xtensa para ESP32-S3) — instalado via EIM
- Extensão **Wokwi for VS Code** (qualquer licença, inclusive Community)
- Python 3.10+ com TensorFlow, NumPy, Matplotlib (para os notebooks)

### Treinamento (opcional — modelos já estão commitados)

```bash
jupyter notebook tflite_hello_world_training.ipynb
jupyter notebook light_classifier_training.ipynb
```

Os notebooks geram o `.tflite` int8 e o convertem para `model.cc`
via `tflite_to_cc.py`.

### Build + simulação

```powershell
# 1. ativar o ambiente ESP-IDF na sessão atual
. C:\esp\v5.4.4\esp-idf\export.ps1

# 2. entrar na pasta do projeto
cd hello_world_tflite     # ou light_classifier_tflite

# 3. configurar e compilar
idf.py set-target esp32s3
idf.py build

# 4. abrir wokwi.toml no VS Code e clicar em ▶ "Start Simulator"
```

> **Atenção** — após o primeiro `idf.py build`, edite a linha 135 de
> `managed_components/espressif__esp-tflite-micro/CMakeLists.txt` para
> comentar `target_compile_options(... -DESP_NN)`. Sem isso, o Wokwi
> trava com `Guru Meditation Error: IllegalInstruction` porque ele não
> emula a extensão PIE/SIMD do Xtensa LX7. Em chip real, mantenha o
> ESP-NN ligado para ganhar 2-5× de inferência.

A correção está documentada em detalhes na seção 5.4 do `relatorio.md`.

## Documento de análise

O `relatorio.md` (e suas exportações `.docx`/`.pdf`) cobre, em 8 seções:

1. Objetivo
2. Pipeline implementado (treino → conversão → deploy)
3. Comparação Float vs Int8 (MAE, RMSE, tamanho)
4. Arquitetura do firmware (`setup`, `loop`, quantização manual)
5. Observações importantes (op resolver seletivo, tensor arena estática,
   `alignas(8)`, ESP-NN incompatível com Wokwi, target esp32 vs esp32s3)
6. Resultado da execução no Wokwi
7. Conclusões
8. **Extra — Light Classifier** (decisões de sensor, dataset, reuso da
   arquitetura, tabela de resultados)

## Resultados

### Hello World (saída no serial)

```
x_value: 0.000000, y_value: 0.054327
x_value: 1.570796, y_value: 0.714014   ← π/2, real ≈ 1.0
x_value: 3.141593, y_value: 0.116415   ← π,   real ≈ 0.0
x_value: 4.712389, y_value: -0.481183  ← 3π/2, real ≈ -1.0
```

MAE int8 medido no notebook: **0,028**.

### Light Classifier (saída no serial)

| ADC raw | norm  | lux_score | classe       |
| ------- | ----- | --------- | ------------ |
| 111     | 0.027 | 0.057     | ESCURO       |
| 1179    | 0.288 | 0.288     | PENUMBRA     |
| 2468    | 0.603 | 0.572     | CLARO        |
| 3658    | 0.893 | 0.932     | MUITO_CLARO  |
