import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
import io
from groq import Groq
import base64
from huggingface_hub import InferenceClient
import time

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set GEMINI_API_KEY in .env file")
    st.stop()

if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY in .env file")
    st.stop()

if not HF_API_KEY:
    st.error("Please set HF_API_KEY in .env file")
    st.stop()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# Configure HuggingFace Client
hf_client = InferenceClient(
    provider="together",
    api_key=HF_API_KEY
)

# Define available models
GEMINI_MODELS = {
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite-preview-02-05",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 1.5 Flash 8B": "gemini-1.5-flash-8b"
}

LLAMA_MODELS = {
    "Llama 3.2 90B Vision": "llama-3.2-90b-vision-preview",
    "Llama 3.2 11B Vision": "llama-3.2-11b-vision-preview"
}

# Configure Streamlit page
st.set_page_config(
    page_title="Math Question Solver",
    page_icon="📝",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .model-selector {
        margin-bottom: 20px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

def process_image(image):
    """Process image to ensure RGB format and reasonable size"""
    try:
        # Convert to RGB if image is RGBA
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Resize image if too large while maintaining aspect ratio
        max_size = 1600
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    try:
        if image is None:
            return None
            
        processed_image = process_image(image)
        if processed_image is None:
            return None
            
        buffered = io.BytesIO()
        processed_image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def get_gemini_response(input_image, model_name):
    """Extract math question from image using selected Gemini model"""
    try:
        if input_image is None:
            return None
            
        processed_image = process_image(input_image)
        if processed_image is None:
            return None
            
        model = genai.GenerativeModel(model_name)
        prompt = "Extract the mathematical question from this image. Include all relevant details and options if present."
        response = model.generate_content([prompt, processed_image])
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        return None

def get_llama_response(input_image, model_name):
    """Extract math question from image using selected Llama model"""
    try:
        base64_image = encode_image_to_base64(input_image)
        if base64_image is None:
            return None
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the mathematical question from this image. Include all relevant details and options if present."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Llama API Error: {str(e)}")
        return None

def get_solution(question):
    """Get solution from Deepseek-R1 model with retry mechanism"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""Please solve this math question step by step and provide a clear explanation:
                    
{question}

Please format your answer as follows:
1. give me the accurate and correct answer with simple steps of explanation 
5. If it's a multiple choice question, clearly indicate which option is correct and why"""
                }
            ]
            
            completion = hf_client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                st.error(f"Error getting solution after {max_retries} attempts: {str(e)}")
                return None

def load_image():
    """Image loader and processor"""
    uploaded_file = st.file_uploader(
        "Upload an image containing a math question", 
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG"
    )
    if uploaded_file is not None:
        try:
            image_data = uploaded_file.read()
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None
    return None

def main():
    st.title("📝 Math Question Solver")
    st.markdown("---")

    # Sidebar for features and model selection
    st.sidebar.title("Features")
    st.sidebar.markdown("""
    - Upload images containing math questions
    - Extract questions using AI vision models
    - Get detailed solutions with step-by-step explanations
    - Download extracted questions and solutions
    - Support for multiple image formats
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Model Selection")
    
    # Model type selection (Gemini or Llama)
    model_type = st.sidebar.radio(
        "Select Model Type",
        ["Gemini", "Llama"],
        help="Choose between Gemini and Llama model families"
    )
    
    # Model selection based on type
    if model_type == "Gemini":
        selected_model_name = st.sidebar.selectbox(
            "Select Gemini Model",
            options=list(GEMINI_MODELS.keys()),
            index=0,
            help="Choose the Gemini model you want to use for extraction",
            key="gemini_model"
        )
        selected_model = GEMINI_MODELS[selected_model_name]
    else:
        selected_model_name = st.sidebar.selectbox(
            "Select Llama Model",
            options=list(LLAMA_MODELS.keys()),
            index=0,
            help="Choose the Llama model you want to use for extraction",
            key="llama_model"
        )
        selected_model = LLAMA_MODELS[selected_model_name]
    
    # Display selected model
    st.sidebar.markdown(f"**Selected Model:** {selected_model_name}")
    st.sidebar.markdown("---")

    # Create three columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### Upload Image")
        image = load_image()
        if image:
            processed_image = process_image(image)
            if processed_image:
                st.image(processed_image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Extract and Solve", type="primary"):
                    with st.spinner(f"Processing with {selected_model_name}..."):
                        try:
                            with col2:
                                st.markdown("### Extracted Question")
                                
                                # Use appropriate model based on selection
                                if model_type == "Gemini":
                                    question = get_gemini_response(processed_image, selected_model)
                                else:
                                    question = get_llama_response(processed_image, selected_model)
                                
                                if question:
                                    st.write(question)
                                    st.download_button(
                                        label="Download Question",
                                        data=question,
                                        file_name="extracted_question.txt",
                                        mime="text/plain"
                                    )
                                    
                                    with col3:
                                        st.markdown("### Solution")
                                        with st.spinner("Generating solution..."):
                                            solution = get_solution(question)
                                            if solution:
                                                st.write(solution)  # Changed from st.markdown to st.write
                                                st.download_button(
                                                    label="Download Solution",
                                                    data=solution,
                                                    file_name="solution.txt",
                                                    mime="text/plain"
                                                )
                                            else:
                                                st.error("Failed to generate solution. Please try again.")
                                else:
                                    st.error("Failed to extract question. Please try again with a different image or model.")
                        except Exception as e:
                            st.error(f"Error during processing: {str(e)}")

    # App info
    with st.expander("About"):
        st.markdown("""
            # Math Question Solver
            
            This app helps you solve math problems in three steps:
            1. Upload an image containing a math question
            2. Extract the question using AI vision models
            3. Generate a detailed solution using Deepseek-R1
            
            ## Available Models
            
            ### Vision Models (for question extraction):
            
            **Gemini Models:**
            - Gemini 2.0 Flash: Latest version, optimized for quick responses
            - Gemini 2.0 Flash Lite: Lightweight version of 2.0 Flash
            - Gemini 1.5 Flash: Balanced performance and speed
            - Gemini 1.5 Flash 8B: Efficient 8B parameter version
            
            **Llama Models:**
            - Llama 3.2 90B Vision: Large vision model with 90B parameters
            - Llama 3.2 11B Vision: Efficient vision model with 11B parameters
            
            ### Solution Model:
            - Deepseek-R1: Advanced math problem-solving model
            
            ## Supported Image Formats
            - PNG
            - JPG/JPEG
            
            Note: Images are automatically processed for optimal performance.
        """)

if __name__ == "__main__":
    main()
