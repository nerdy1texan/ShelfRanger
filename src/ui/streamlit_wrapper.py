"""
Streamlit wrapper for ShelfRanger with proper CUDA handling.
This script handles CUDA initialization issues in Streamlit.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_cuda_environment():
    """Setup environment variables for CUDA compatibility in Streamlit."""
    
    # Set PyTorch environment variables
    os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
    
    # CUDA compatibility settings for Streamlit
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # RTX 3060 architecture
    
    # Force CUDA initialization in controlled manner
    try:
        import torch
        if torch.cuda.is_available():
            # Initialize CUDA context properly
            device = torch.cuda.current_device()
            torch.cuda.empty_cache()
            print(f"✅ CUDA initialized successfully on device {device}")
            return True
    except Exception as e:
        print(f"⚠️ CUDA initialization issue: {e}")
        print("🔄 Falling back to CPU mode for Streamlit compatibility")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return False

def run_streamlit():
    """Run the Streamlit app with proper CUDA setup."""
    
    # Setup CUDA environment
    cuda_available = setup_cuda_environment()
    
    # Get the UI script path
    ui_script = Path(__file__).parent / "gpu_streamlit_app.py"
    
    if not ui_script.exists():
        print(f"❌ UI script not found: {ui_script}")
        return
    
    # Run Streamlit with proper environment
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_script)]
    
    print(f"🚀 Starting Streamlit with {'GPU' if cuda_available else 'CPU'} support...")
    print(f"📂 Script: {ui_script}")
    print(f"🌐 Will be available at: http://localhost:8501")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    run_streamlit() 