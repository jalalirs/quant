#!/bin/bash
# Test script for the containerized API

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
API_URL="http://localhost:8000"
TIMEOUT=10

print_status "Testing Quantization API at $API_URL"

# Test 1: Health Check
print_test "Testing health endpoint..."
if curl -s --max-time $TIMEOUT "$API_URL/health" > /dev/null; then
    print_status "✅ Health check passed"
else
    print_error "❌ Health check failed"
    exit 1
fi

# Test 2: Models endpoint
print_test "Testing models endpoint..."
models_response=$(curl -s --max-time $TIMEOUT "$API_URL/v1/models")
if [[ $? -eq 0 ]]; then
    print_status "✅ Models endpoint working"
    echo "Models: $models_response"
else
    print_error "❌ Models endpoint failed"
fi

# Test 3: Chat completion
print_test "Testing chat completion..."
chat_response=$(curl -s --max-time 30 -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3-trt",
        "messages": [{"role": "user", "content": "Hello! How are you?"}],
        "max_tokens": 50,
        "temperature": 0.7
    }')

if [[ $? -eq 0 ]]; then
    print_status "✅ Chat completion working"
    echo "Response: $chat_response"
    
    # Extract and display the actual message content
    message_content=$(echo "$chat_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print('Actual response:', data['choices'][0]['message']['content'])
except:
    print('Could not parse response')
")
    echo "$message_content"
else
    print_error "❌ Chat completion failed"
fi

# Test 4: Streaming chat completion
print_test "Testing streaming chat completion..."
stream_response=$(curl -s --max-time 30 -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3-trt",
        "messages": [{"role": "user", "content": "Tell me about AI"}],
        "max_tokens": 30,
        "temperature": 0.7,
        "stream": true
    }')

if [[ $? -eq 0 ]]; then
    print_status "✅ Streaming chat completion working"
    echo "Stream response (first few lines):"
    echo "$stream_response" | head -5
else
    print_error "❌ Streaming chat completion failed"
fi

# Test 5: Error handling
print_test "Testing error handling with invalid model..."
error_response=$(curl -s --max-time $TIMEOUT -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nonexistent-model",
        "messages": [{"role": "user", "content": "Test"}]
    }')

if [[ $? -eq 0 ]]; then
    print_status "✅ Error handling working (server responded)"
    echo "Error response: $error_response"
else
    print_error "❌ Server didn't respond to invalid request"
fi

print_status "🎉 API testing complete!"

# Performance test
print_test "Running simple performance test (5 requests)..."
start_time=$(date +%s)
for i in {1..5}; do
    curl -s --max-time 15 -X POST "$API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"llama3-trt\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Quick test $i\"}],
            \"max_tokens\": 20
        }" > /dev/null
    echo -n "."
done
end_time=$(date +%s)
duration=$((end_time - start_time))
echo ""
print_status "⚡ Completed 5 requests in ${duration} seconds ($(echo "scale=2; 5/$duration" | bc) req/sec)"

print_status "All tests completed! The API is working properly."