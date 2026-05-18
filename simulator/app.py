from pathlib import Path
import numpy as np # type: ignore
import streamlit as st # type: ignore
import cv2 # type: ignore


# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_NAME = BASE_DIR / "default_images" / "power_input.jpg"
DEFAULT_TONE_IMAGE_NAME = BASE_DIR / "default_images" / "hdr_input.hdr"


def apply_gamma_correction(img: np.ndarray, gamma: float, c: float = 255.0) -> np.ndarray:
	# basic power law
	img_f = img.astype(np.float64)
	img_f = c * (img_f / c) ** gamma
	return np.clip(img_f, 0, c).astype(np.uint8)
#

def apply_reinhard(img: np.ndarray, exposure: float = 1.0, gamma: float = 1.0 / 2.2) -> np.ndarray:
	# accept HDR floats unchanged, otherwise scale uint8 -> [0,1]
	img = img.astype(np.float64)

	# apply exposure on linear radiance
	img *= float(exposure)

	# Reinhard operator per-channel: L_d = L / (1 + L)
	ldr = img / (img + 1.0)

	# apply display gamma and convert to uint8
	out = np.clip(ldr ** float(gamma), 0.0, 1.0)
	return (out * 255.0).astype(np.uint8)
#

def apply_hable(img: np.ndarray, exposure: float = 1.0, gamma: float = 1.0 / 2.2) -> np.ndarray:
	def hable_operator(x: np.ndarray) -> np.ndarray:
		A = 0.15
		B = 0.50
		C = 0.10
		D = 0.20
		E = 0.02
		F = 0.30
		# Equation: ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F
		return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

	# accept HDR floats unchanged, otherwise scale uint8 -> [0,1]
	x = img.astype(np.float64)

	# exposure
	x *= float(exposure)
	
	# filmic curve
	x = hable_operator(x)

	# apply display gamma and convert to uint8
	out = np.clip(x ** float(gamma), 0.0, 1.0)
	return (out * 255.0).astype(np.uint8)
#

@st.cache_data()
def load_default_gamma_correction_image() -> np.ndarray:
	default_path = DEFAULT_IMAGE_NAME

	img = cv2.imread(str(default_path), cv2.IMREAD_COLOR)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#

@st.cache_data()
def load_default_tone_mapping_image() -> np.ndarray:
	default_path = DEFAULT_TONE_IMAGE_NAME

	# HDR read
	img = cv2.imread(str(default_path), -1)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#

def image_to_display_columns(original: np.ndarray, corrected: np.ndarray) -> None:
	left, right = st.columns(2)

	def _print_original(img: np.ndarray) -> np.ndarray:
		if np.issubdtype(img.dtype, np.floating):
			if img.size == 0:
				return img.astype(np.float32)
		
			m = float(np.nanmax(img))
			if m > 1.0:
				p = float(np.nanpercentile(img, 99.5))
				
				if p <= 0.0:
					p = m
				return np.clip(img / float(p), 0.0, 1.0).astype(np.float32)
			
			return np.clip(img, 0.0, 1.0).astype(np.float32)
		# integers (uint8) are already displayable
		return img

	with left:
		st.subheader("Original (as LDR)")
		st.image(_print_original(original), width=900)

	with right:
		st.subheader("Processed Output")
		st.image(corrected, width=900)
#

def render_simulation(title: str, original: np.ndarray, corrected: np.ndarray) -> None:
	st.subheader(title)
	image_to_display_columns(original, corrected)
#

def main() -> None:
	st.set_page_config(page_title="Gamma Correction and Tone Mapping", layout="wide")

	gamma = 1
	c: float = 255.0
	exposure = 1.5
	tone_mode = "Reinhard"

	st.markdown(
		"""
		<style>
		.block-container {
			padding-top: 2rem;
			padding-bottom: 2rem;
		}
		.hero {
			padding: 1.25rem 1.5rem;
			border-radius: 1rem;
			background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #374151 100%);
			color: white;
			margin-bottom: 1.25rem;
		}
		.hero p {
			margin-bottom: 0;
			opacity: 0.9;
		}
		</style>
		""",
		unsafe_allow_html=True
	)

	st.markdown(
		f"""
		<div class="hero">
			<h1>{"Gamma Correction and Tone Mapping"}</h1>
		</div>
		""",
		unsafe_allow_html=True,
	)

	with st.sidebar:
		st.header("Controls")
		uploaded_file = st.file_uploader("Upload a custom image", type=["png", "jpg", "jpeg", "webp"])

		if uploaded_file is not None:
			file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
			bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
			
			rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
			orig_for_gamma = rgb
			orig_for_tone = rgb
		else:
			orig_for_gamma = load_default_gamma_correction_image()
			orig_for_tone = load_default_tone_mapping_image()

	tabs = st.tabs(["Gamma correction", "Tone mapping"])

	with tabs[0]: # Gamma correction tab
		gamma = st.slider(
			"Gamma (ɣ)", 
			min_value=0.05, 
			max_value=5.00, 
			value=float(gamma), 
			step=0.01
		)
		c = st.slider(
			"Scale (c)", 
			min_value=1.0, 
			max_value=255.0, 
			value=float(c), 
			step=1.0
		)
		
		st.caption("Gamma values below 1 brighten shadows, whereas values above 1 darken the image.")
		
		render_simulation(
			"Gamma correction preview",
			orig_for_gamma,
			apply_gamma_correction(orig_for_gamma, gamma=gamma, c=c)
		)

	with tabs[1]: # Tone mapping tab
		tone_mode = st.radio("Tone mapping operator", ["Reinhard", "Hable"], index=0)
		
		exposure = st.slider(
			"Exposure", 
			min_value=0.10, 
			max_value=4.00, 
			value=float(exposure), 
			step=0.01
		)
		tone_gamma = st.slider(
			"Gamma (display)", 
			min_value=0.05, 
			max_value=5.00, 
			value=float(1.0 / 2.2), 
			step=0.01
		)
		
		st.caption("Reinhard tone mapping compresses highlights while preserving detail in bright regions. Use Exposure and Gamma to control appearance.")
		
		if tone_mode == "Reinhard":
			img_processed = apply_reinhard(orig_for_tone, exposure=exposure, gamma=tone_gamma)
		else: # Hable
			img_processed = apply_hable(orig_for_tone, exposure=exposure, gamma=tone_gamma)
		
		render_simulation(
			"Reinhard tone mapping preview",
			orig_for_tone,
			img_processed
		)


if __name__ == "__main__":
	main()
