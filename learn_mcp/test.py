#!/usr/bin/env python3
"""Test script for VLM LM Studio MCP server"""

import sys
import os
import json

# Add current directory to path to import vlm_lmstudio
sys.path.insert(0, os.path.dirname(__file__))

from vlm_lmstudio import describe, detect_person, ocr, format_query

# Test image path
IMAGE_PATH = "D:/speech_research/language_based_audio_retrieval/image.png"

def test_tool(tool_func, tool_name, image_path, *args):
    """Test a specific tool"""
    
    # Check if image exists first
    if not os.path.exists(image_path):
        print(f"\n{'='*60}")
        print(f"Testing: {tool_name}")
        print(f"{'='*60}")
        print(f"ERROR: Image file not found at {image_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing: {tool_name}")
    print(f"Image: {image_path}")
    if args:
        print(f"Args: {args}")
    print(f"{'='*60}")
    
    try:
        if args:
            result = tool_func(*args, image_path)
        else:
            result = tool_func(image_path)
        
        # Pretty print if result is a dict
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(f"Result:\n{result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("VLM MCP Server Direct Test")
    print(f"Image: {IMAGE_PATH}")
    
    # Test basic tools
    test_tool(ocr, "OCR", IMAGE_PATH)
    test_tool(describe, "Describe", IMAGE_PATH)
    test_tool(detect_person, "Detect Person", IMAGE_PATH)
    test_tool(present_checker, "Audio information retrieval", IMAGE_PATH)
    test_tool(format_query, "Format Query", IMAGE_PATH, "block diagram ")
    
    print(f"\n{'='*60}")
    print("Tests completed!")
    print(f"{'='*60}")
