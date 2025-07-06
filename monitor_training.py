#!/usr/bin/env python3
"""
Simple script to monitor training progress
"""
import os
import time

def monitor_training():
    """Monitor the training directory for progress"""
    model_dir = "star-gpt2-finetuned"
    
    if not os.path.exists(model_dir):
        print("Training directory not found. Training may not have started yet.")
        return
    
    print("Monitoring training progress...")
    print("=" * 50)
    
    while True:
        try:
            # List files in the training directory
            files = os.listdir(model_dir)
            files.sort()
            
            print(f"\nFiles in {model_dir}:")
            for file in files:
                file_path = os.path.join(model_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file}: {size:,} bytes")
            
            # Check for completion indicators
            if "pytorch_model.bin" in files or "model.safetensors" in files:
                print("\n🎉 Training appears to be complete!")
                print("You can now test the model with:")
                print("  python test_model.py")
                print("  python generate.py")
                print("  python app.py")
                break
            
            print("\n⏳ Training still in progress...")
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"Error monitoring: {e}")
            break

if __name__ == "__main__":
    monitor_training()
