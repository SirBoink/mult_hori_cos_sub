# Human data

Sequential reaching of 20 participants (fingertip, 100 Hz), from

- Cuevas Rivera, D., Szpilska, A. & Kiebel, S. *Sequential reaching movements with coarticulation* [dataset], Figshare.
- Cuevas Rivera, D. & Kiebel, S. J. *Behavioral evidence for the hierarchical execution of sequential movements*,
  Communications Psychology (2026), doi:10.1038/s44271-026-00436-5; code and model outputs doi:10.6084/m9.figshare.31169632.

Both are shared under CC BY 4.0. Files are unmodified.

| File | Content | sha256 |
|---|---|---|
| `bids_data.zip` | the BIDS dataset (motion files, experiment sheet, README) | `4e64e69c4b85ceccfca7e328d1971e79d8bb435281925ff34308ea6af5afd236` |
| `published_models/data_chalf.csv` | per-trial coarticulation and curvature of the participants | `3ca2e32e3bb6c4f91f4ceb09ad87f6ca535244250c7d3570bd3760b7d5045a57` |
| `published_models/vpsoc_chalf.csv` | the same measures for the flat via-point model | `7448973d04c10e2ae99bef66fb0d2487f3aef6627962954b42cd54a7a03f6ab7` |
| `published_models/hiseq_chalf.csv` | the same measures for the hierarchical model | `4fe1e38698203e6d2709bcf2a463da30a69bec2641cc41f187fd0d62d8ea3ad1` |
| `published_models/src/` | the authors' analysis code needed for `analyses.model_parts` (which model covers each participant and sequence), in the package layout its imports expect | `analyses.py` `350a9c8e…` |

`analyze_humans.py` uses `model_parts` to label each participant's sequence as hierarchical or flat.
