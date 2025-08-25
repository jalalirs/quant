"""
OpenAI-compatible API server for quantized models
"""

import json
import time
import asyncio
import logging
import threading
from typing import Dict, List, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)

# OpenAI-compatible API models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 100
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]

class ServerOpenAI:
    """OpenAI-compatible API server"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get('host', '0.0.0.0')
        self.port = config.get('port', 6001)
        self.model_name = config.get('model_name', 'llama3-trt')
        
        # Get quantizer from config
        self.quantizer = config.get('quantizer')
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Quantized Model Server")
        self._setup_routes()
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'ServerOpenAI':
        """Load OpenAI server from configuration"""
        return cls(config)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": "llama3-trt",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local"
                    }
                ]
            }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                # Convert messages to prompt
                prompt = self._messages_to_prompt(request.messages)
                
                if request.stream:
                    return StreamingResponse(
                        self._generate_stream(prompt, request),
                        media_type="text/plain"
                    )
                else:
                    # Handle both traditional quantizer and optimized loader
                    if hasattr(self.quantizer, 'generate_text'):
                        response_text = self.quantizer.generate_text(
                            prompt, 
                            max_new_tokens=request.max_tokens,
                            temperature=request.temperature
                        )
                    elif hasattr(self.quantizer, 'generate'):
                        response_text = self.quantizer.generate(
                            prompt,
                            max_new_tokens=request.max_tokens,
                            temperature=request.temperature
                        )
                    else:
                        raise RuntimeError("Quantizer does not support text generation")
                    
                    return ChatCompletionResponse(
                        id=f"chatcmpl-{int(time.time())}",
                        created=int(time.time()),
                        model=request.model,
                        choices=[{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }]
                    )
                    
            except Exception as e:
                logger.error(f"Chat completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "model": self.model_name}
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert OpenAI messages to prompt format"""
        prompt_parts = []
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"Human: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        prompt_parts.append("Assistant: ")
        return "\n".join(prompt_parts)
    
    async def _generate_stream(self, prompt: str, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        # Handle both traditional quantizer and optimized loader
        if hasattr(self.quantizer, 'generate_text'):
            response_text = self.quantizer.generate_text(
                prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
            )
        elif hasattr(self.quantizer, 'generate'):
            response_text = self.quantizer.generate(
                prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
            )
        else:
            raise RuntimeError("Quantizer does not support text generation")
        
        # Simulate streaming by yielding chunks
        words = response_text.split()
        for i, word in enumerate(words):
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " " if i < len(words) - 1 else word},
                    "finish_reason": None if i < len(words) - 1 else "stop"
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.05)  # Small delay for streaming effect
        
        yield "data: [DONE]\n\n"
    
    def start(self):
        """Start the server (blocking)"""
        logger.info(f"Starting OpenAI-compatible server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)
    
    def start_async(self):
        """Start the server in async mode"""
        logger.info(f"Starting OpenAI-compatible server on {self.host}:{self.port}")
        
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        config = uvicorn.Config(
            self.app, 
            host=self.host, 
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Run server
        loop.run_until_complete(server.serve())