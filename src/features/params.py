"""Single source of truth for the front-end parameters (interim spec, section 3)."""

SR = 16_000
CLIP_SECONDS = 4.0
N_SAMPLES = int(SR * CLIP_SECONDS)  # 64,000

# Log-mel for CNN / CRNN / MobileNetV3
N_FFT = 1024
HOP = 320
N_MELS = 128
FMIN = 20.0
FMAX = 8000.0
N_FRAMES = 1 + N_SAMPLES // HOP  # 201 with center=True
AMIN = 1e-10  # power floor before dB; 10*log10(1e-10) = -100 dB

# AST front-end (matches transformers.ASTFeatureExtractor with max_length=400)
AST_MEL_BINS = 128
AST_FRAMES = 400
AST_MEAN = -4.2677393
AST_STD = 4.5689974

N_FOLDS = 10
N_CLASSES = 10
