# Project Maintenance & Workflow Log

## System Workflow
1. **User Interaction**: Frontend (React + Vite) sends user input and `user_id` to FastAPI backend.
2. **Context Retrieval**: Backend extracts past conversation history and user persona traits from SQLite.
3. **Agent Decision**: LangGraph queries Ollama with custom system prompts containing user memory[cite: 1].
4. **Memory Evolution**: Backend parses messages for personal preferences and updates the profile automatically[cite: 1].
5. **Feedback Loop**: Frontend rating (up/down vote) updates confidence scores in the user model[cite: 1].

## Change Log
- **v1.0.0**: Initial baseline. Dual-layer SQLite memory engine, FastAPI agent loop, React dark-mode dashboard.

## Common Issues & Troubleshooting
- **Ollama Connection Refused**: Verify Ollama is running (`ollama serve`) and model is pulled (`ollama list`)[cite: 1].
- **CORS Errors**: Check `ORIGINS` inside `backend/app/config.py` matches Vite's dev port (default `5173`)[cite: 1].
- **Slow Inference**: Ensure your quantized GGUF fits in GPU VRAM or reduce context length in `agent.py`.