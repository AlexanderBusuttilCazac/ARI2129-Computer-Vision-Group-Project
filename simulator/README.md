# Interactive Gamma & Tone Processing Simulator

## Run

```bash
pip install -r requirements.txt
streamlit run simulator/app.py
```

## Features

- Gamma correction (encode/decode and custom power law)
- sRGB piecewise transfer
- Log-like encoding approximation
- Rec.2020-style power transfer approximation
- Global HDR tone mapping (Reinhard)
- Interactive side-by-side visualization and histograms
