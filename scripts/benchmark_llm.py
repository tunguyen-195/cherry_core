import sys
from pathlib import Path
import time
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Force verbose output from llama.cpp
os.environ["LLAMA_CPP_LIB_VERBOSE"] = "1"

from infrastructure.adapters.llm.llamacpp_adapter import LlamaCppAdapter

def benchmark():
    print("🚀 STARTING LLAMA.CPP BENCHMARK...")
    print(f"Checking for GPU support...")
    
    try:
        from llama_cpp import Llama
        print("✅ llama-cpp-python imported.")
    except ImportError:
        print("❌ llama-cpp-python NOT installed.")
        return

    engine = LlamaCppAdapter()
    
    # Override for benchmark
    engine.context_window = 2048 
    
    start_load = time.time()
    if not engine.load():
        print("❌ Model Load Failed.")
        return
    load_time = time.time() - start_load
    print(f"⏱️ Model Load Time: {load_time:.2f}s")
    
    # Access internal llm object to check metadata
    if hasattr(engine.llm, "n_gpu_layers"):
         print(f"ℹ️ Configured GPU Layers: {engine.llm.n_gpu_layers}")
    
    prompt = """<|im_start|>user
Hãy giới thiệu ngắn gọn về bản thân bạn trong 50 từ.<|im_end|>
<|im_start|>assistant
"""
    
    print("\n🧪 Running Inference (Generation)...")
    start_gen = time.time()
    output = engine.generate(prompt, max_tokens=100)
    gen_time = time.time() - start_gen
    
    print(f"\n📝 Output: {output}")
    print(f"\n⚡ Inference Time: {gen_time:.2f}s")
    
    # Crude estimation
    tokens = len(output.split()) * 1.3 # approx
    tps = tokens / gen_time
    print(f"🏎️ Estimated Speed: ~{tps:.2f} tokens/sec")

if __name__ == "__main__":
    benchmark()
