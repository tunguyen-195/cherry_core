try:
    from llama_cpp import LlamaGrammar
    print("✅ llama_cpp imported")
except ImportError:
    print("❌ llama_cpp not installed")
    exit(1)

grammar_path = "prompts/grammars/json_schema.gbnf"

try:
    print(f"📖 Loading grammar from: {grammar_path}")
    grammar = LlamaGrammar.from_file(grammar_path)
    print("✅ Grammar loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load grammar: {e}")
