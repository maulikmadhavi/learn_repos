import asyncio
import aiohttp
import time
import json
from pathlib import Path
import os
import argparse
from tqdm import tqdm
import statistics
import random

IMG_DIR = "/mnt/d/custom_tools/vlm_model_serve/bentoml/val2017/val2017"    

async def make_request(session, url, image_path, text, request_id):
    """Make a single request to the BLIP image captioning service."""
    start_time = time.time()
    try:
        data = aiohttp.FormData()
        data.add_field('img', 
                      open(image_path, 'rb'),
                      filename=os.path.basename(image_path),
                      content_type='image/jpeg')
        
        if text:
            data.add_field('txt', text, content_type='application/json')
        
        # Set timeout to avoid hanging requests
        async with session.post(url, data=data, timeout=60) as response:
            if response.status == 200:
                result = await response.json()
                elapsed = time.time() - start_time
                return {
                    'request_id': request_id,
                    'status': 'success',
                    'elapsed_time': elapsed,
                    'time_to_first_token': result.get('time_to_first_token'),
                    'caption': result.get('caption'),
                    'http_status': response.status,
                    'image_path': image_path
                }
            else:
                error_text = await response.text()
                return {
                    'request_id': request_id,
                    'status': 'error',
                    'elapsed_time': time.time() - start_time,
                    'http_status': response.status,
                    'error': error_text,
                    'image_path': image_path
                }
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        return {
            'request_id': request_id,
            'status': 'exception',
            'elapsed_time': elapsed,
            'error': f"Request timed out after {elapsed:.2f} seconds",
            'image_path': image_path
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'request_id': request_id,
            'status': 'exception',
            'elapsed_time': elapsed,
            'error': str(e),
            'image_path': image_path
        }

async def run_concurrent_requests(url, text, num_requests, concurrency, use_random_images=True):
    """Run multiple requests with a concurrency limit."""
    
    # Check if IMG_DIR exists and has images
    if not os.path.exists(IMG_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")
    
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise FileNotFoundError(f"No image files found in {IMG_DIR}")
    
    print(f"Found {len(image_files)} images in directory")
    
    # Configure client session with proper connection settings
    tcp_connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=concurrency,
        force_close=False,
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(total=120)  # 2-minute total timeout
    
    async with aiohttp.ClientSession(
        connector=tcp_connector, 
        timeout=timeout,
        raise_for_status=False
    ) as session:
        tasks = []
        for i in range(num_requests):
            # Select either a random image each time or reuse the same image
            if use_random_images:
                image_path = os.path.join(IMG_DIR, random.choice(image_files))
            else:
                image_path = os.path.join(IMG_DIR, image_files[0])
                
            task = asyncio.create_task(make_request(session, url, image_path, text, i))
            tasks.append(task)
        
        # Use tqdm to show progress
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                print(f"Task exception: {e}")
        
        return results

def analyze_results(results):
    """Analyze and print statistics from the test results."""
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    # Calculate statistics for successful requests
    if successful:
        elapsed_times = [r['elapsed_time'] for r in successful]
        ttft_times = [r['time_to_first_token'] for r in successful if r.get('time_to_first_token') is not None]
        
        print("\n===== RESULTS =====")
        print(f"Total requests: {len(results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.2f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.2f}%)")
        
        print("\n===== LATENCY (seconds) =====")
        print(f"Min: {min(elapsed_times):.4f}")
        print(f"Max: {max(elapsed_times):.4f}")
        print(f"Mean: {statistics.mean(elapsed_times):.4f}")
        print(f"Median: {statistics.median(elapsed_times):.4f}")
        if len(elapsed_times) > 1:
            print(f"StdDev: {statistics.stdev(elapsed_times):.4f}")
            
        if ttft_times:
            print("\n===== TIME TO FIRST TOKEN (seconds) =====")
            print(f"Min: {min(ttft_times):.4f}")
            print(f"Max: {max(ttft_times):.4f}")
            print(f"Mean: {statistics.mean(ttft_times):.4f}")
            print(f"Median: {statistics.median(ttft_times):.4f}")
            if len(ttft_times) > 1:
                print(f"StdDev: {statistics.stdev(ttft_times):.4f}")
    
    # Print error summary
    if failed:
        error_types = {}
        for f in failed:
            error_type = f.get('http_status', 'exception')
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        print("\n===== ERRORS =====")
        for error_type, count in error_types.items():
            print(f"{error_type}: {count}")
            
        # Print a few example errors for debugging
        print("\nExample errors:")
        for i, f in enumerate(failed[:3]):  # Show first 3 errors
            print(f"Error {i+1}: {f.get('error', 'Unknown error')}")

def save_results(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test BLIP image captioning service with concurrent requests')
    parser.add_argument('--url', default='http://localhost:3000/generate', help='Service URL')
    parser.add_argument('--text', default='describe the image', help='Text prompt')
    parser.add_argument('--requests', type=int, default=200, help='Number of requests to make')
    parser.add_argument('--concurrency', type=int, default=50, help='Maximum concurrent requests')
    parser.add_argument('--output', default='blip_benchmark_results.json', help='Output file for results')
    parser.add_argument('--use-single-image', action='store_true', help='Use the same image for all requests')
    parser.add_argument('--image-dir', help='Directory containing images (overrides IMG_DIR)')
    args = parser.parse_args()
    
    # Override IMG_DIR if provided
    if args.image_dir:
        IMG_DIR = args.image_dir
        print(f"Using image directory: {IMG_DIR}")
    
    # Check the URL format
    if not args.url.startswith(('http://', 'https://')):
        args.url = f"http://{args.url}"
        print(f"Updated URL to include protocol: {args.url}")
    
    print(f"Starting benchmark with {args.requests} requests, {args.concurrency} max concurrent...")
    print(f"URL: {args.url}")
    print(f"Text prompt: {args.text}")
    print(f"Using {'single' if args.use_single_image else 'random'} images")
    
    try:
        start_time = time.time()
        results = asyncio.run(run_concurrent_requests(
            args.url, args.text, args.requests, args.concurrency, 
            not args.use_single_image  # If use_single_image is True, use_random_images should be False
        ))
        total_time = time.time() - start_time
        
        print(f"\nTotal benchmark time: {total_time:.2f} seconds")
        print(f"Average throughput: {args.requests / total_time:.2f} requests/second")
        
        analyze_results(results)
        save_results(results, args.output)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nError running benchmark: {e}")