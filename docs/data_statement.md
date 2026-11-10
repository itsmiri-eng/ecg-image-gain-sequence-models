# Data statement

- **Source.** PTB-XL v1.0.3 (Wagner et al., Sci Data 2020; PhysioNet). License: CC BY 4.0. It is open access
  and needs no credentialing.
- **Records.** There are 21,799 clinical 12-lead, 10-s ECGs from 18,869 patients, recorded at
  Physikalisch-Technische Bundesanstalt / Schiller devices in Germany between 1989 and 1996. The cohort here is
  the 21,388 records with at least one diagnostic superclass label.
- **Labels.** These are the five PTB-XL diagnostic superclasses (NORM, MI, STTC, CD, HYP), derived from the
  cardiologist SCP-ECG statements with PTB-XL's own `scp_statements.csv`. Labels overlap: one ECG can have
  several.
- **Splits.** The official stratified folds keep each patient in one fold: 1–8 train (17,084), 9 validation
  (2,146), 10 test (2,158).
- **Images.** Images are synthetic paper-style renderings of the 500 Hz waveforms, made by this project
  (8 px/mm, 25 mm/s, 12 stacked rows, 1 mV calibration pulse per row). They are not scans or photos. No
  identifiers, demographics or diagnoses are printed on the pages.
- **Personal data.** None beyond what PTB-XL publishes: pseudonymised IDs, age, sex, height and weight. None of
  these is a model input. No personal documents are part of the project data.
- **Known biases and limits:**
  - one country and one era;
  - label noise in the original annotations;
  - adaptive page gain in the v1 renderer encodes amplitude, which correlates with hypertrophy (studied
    explicitly; see `docs/ptbxl_fixed_gain_protocol.md`);
  - fold 10 was observed before the model specification.
- **External data (planned).** The planned external set is PhysioNet/CinC Challenge 2021 (CC BY 4.0), with
  labels mapped through PTB-XL+ (CC BY 4.0). See `docs/external_test_plan.md`.
