/*
 * BrainBoard BioAmp EXG Pill firmware reference
 * ----------------------------------------------
 * Reads the analog output of a BioAmp EXG Pill in EOG mode and streams
 * one ADC sample per line over USB serial. This is the host-side
 * counterpart to BioAmpEXGDriver.
 *
 * Wiring:
 *   BioAmp EXG Pill OUT  ──── A0 (analog input)
 *   BioAmp EXG Pill GND  ──── GND
 *   BioAmp EXG Pill VCC  ──── 3.3V (or 5V depending on your board variant)
 *
 *   Electrodes (EOG configuration, single-eye vertical):
 *     IN+ ──── above the eye (or on the eyebrow)
 *     IN- ──── below the eye (cheekbone)
 *     REF ──── on the forehead / behind the ear
 *
 * Sampling: 500 Hz (matches BioAmpConfig.sample_rate_hz default).
 *
 * Output format:
 *   <decimal int>\n
 *   …
 *
 * No bandpass is applied here in firmware. The driver applies its own
 * bandpass (0.5-10 Hz) via scipy on the host side. If you want to offload
 * the filter to the MCU, reuse the EXG_Filter() function from the
 * Upside Down Labs reference repo and replace `raw` below with
 * `EXG_Filter(raw)`.
 *
 * Tested on:
 *   - Maker Uno (ATmega328P)
 *   - ESP32 DevKit (use ADC1 only; ADC2 conflicts with WiFi)
 *
 * If you target ESP32, change `INPUT_PIN` to e.g. 36 (GPIO36 / VP).
 */

const int INPUT_PIN = A0;
const unsigned long SAMPLE_INTERVAL_US = 2000UL;  // 500 Hz

unsigned long next_sample_us = 0;

void setup() {
  Serial.begin(115200);
  // Print a banner so the host can verify the link is alive.
  // The driver tolerates non-numeric lines and skips them.
  Serial.println("BIOAMP READY");

  // ESP32: analogReadResolution(12); analogSetAttenuation(ADC_11db);
  // Uno is 10-bit by default; that's already what BioAmpConfig defaults assume.

  next_sample_us = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - next_sample_us) >= 0) {
    int raw = analogRead(INPUT_PIN);
    Serial.println(raw);
    next_sample_us += SAMPLE_INTERVAL_US;
  }
}
