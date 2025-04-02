import os
import json
import traceback
import argparse
from datetime import datetime
from video_processor import VideoProcessor
import sys
sys.path.append('Yolov5_StrongSORT_OSNet')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process and correct corrupted videos')
    
    parser.add_argument('--input', '-i', type=str, help='Path to video for processing', required=True)
    parser.add_argument('--output', '-o', type=str, help='Output path for processed video')
    parser.add_argument('--output-dir', '-d', type=str, default='video_output', 
                      help='Output directory for all results')
    
    parser.add_argument('--max-frames', type=int, default=None, 
                      help='Maximum number of frames to process (None for all)')
    parser.add_argument('--fps', type=float, default=25.0, 
                      help='Frames per second for output video')
    
    parser.add_argument('--reordering', '-r', type=str, choices=['tracking', 'topological', 'optical_flow', 'feature_matching', 'fused'], 
                  default='tracking', help='Frame reordering method')
    
    parser.add_argument('--outlier-methods', '-m', nargs='+', 
                      default=['zscore', 'pca'], 
                      help='Outlier detection methods')
    
    parser.add_argument('--tracking-confidence', type=float, default=0.8, 
                      help='Confidence threshold for person tracking')
    
    parser.add_argument('--yolo-model', type=str, default='yolov8m.pt', 
                      help='Path to YOLO model for detection')
    
    parser.add_argument('--tracker', type=str, default='deepsort',
                    choices=['botsort', 'bytetrack', 'deepsort', 'strongsort'],
                    help='Choix du tracker : deepsort ou strongsort')
    
    parser.add_argument('--visualize-all', '-v', action='store_true', 
                      help='Generate all visualizations and analyses')
    
    parser.add_argument('--analyze-objects', '-a', action='store_true', 
                      help='Run object analysis on the video')
    
    parser.add_argument('--batch', '-b', nargs='+', 
                      help='Process multiple videos in batch mode')
    
    parser.add_argument('--track-id', type=int, default=None, 
                      help='Target track_id for tracking-based reordering')
    
    return parser.parse_args()

def process_single_video(processor, video_path, output_path, analyze_objects=False):
    """Process a single video"""
    if not os.path.exists(video_path):
        print(f"[ERROR] Video {video_path} does not exist.")
        return False

    print(f"\n--- Processing {video_path} ---")
    metadata = processor.process_video(video_path, output_path)

    print("\nProcessing Metadata:")
    for key, value in metadata.items():
        if key not in ['process_dirs', 'config']:
            print(f"{key}: {value}")

    if analyze_objects and metadata.get('status') == 'success':
        # Obtenir les frames reordonnées si disponibles
        frames = metadata.get('reordered_frames', [])
        
        if isinstance(frames, list) and len(frames) > 0:
            print(f"\nExtracted frames: {len(frames)}")
            print("Analyzing objects in video...")

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            analysis_dir = os.path.join(processor.config['output_root'], 'object_analysis', video_name)
            os.makedirs(analysis_dir, exist_ok=True)

            object_analysis = processor.analyze_video_objects(frames, output_dir=analysis_dir)

            print("\nDetection summary:")
            print(f"Total frames: {object_analysis['total_frames']}")
            print(f"Frames with detections: {object_analysis['frames_with_detections']}")
            print(f"Total detections: {object_analysis['total_detections']}")

            print("\nObject counts:")
            for obj, count in sorted(object_analysis['object_counts'].items(), key=lambda x: x[1], reverse=True):
                print(f"{obj}: {count}")
        else:
            print("[WARNING] No frames available for object analysis.")

    return metadata.get('status') == 'success'

def process_batch(processor, video_paths, output_dir, analyze_objects=False):
    """Process a batch of videos"""
    results = []
    
    # Create a directory for batch results
    batch_dir = os.path.join(output_dir, 'batch', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(batch_dir, exist_ok=True)
    
    # Process each video
    for video_path in video_paths:
        # Define output path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(batch_dir, f'processed_{video_name}.mp4')
        
        # Process the video
        try:
            success = process_single_video(processor, video_path, output_path, analyze_objects)
            
            results.append({
                'input': video_path,
                'output': output_path,
                'success': success,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"[ERROR] Failed to process {video_path}: {e}")
            traceback.print_exc()
            
            results.append({
                'input': video_path,
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Save batch processing summary
    summary_path = os.path.join(batch_dir, 'batch_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display summary
    print("\n=== Batch processing summary ===")
    print(f"Total videos: {len(results)}")
    print(f"Successfully processed videos: {sum(1 for r in results if r.get('success'))}")
    print(f"Failures: {sum(1 for r in results if not r.get('success'))}")
    print(f"Summary saved: {summary_path}")
    
    return results

def main():
    args = parse_arguments()

    # Create processor with args
    processor = VideoProcessor({
    'output_root': args.output_dir,
    'max_frames': args.max_frames,
    'fps': args.fps,
    'reordering_method': args.reordering,
    'outlier_detection_methods': args.outlier_methods,
    'tracking_confidence': args.tracking_confidence,
    'yolo_model_path': args.yolo_model,
    'save_all_visualizations': args.visualize_all,
    'analyze_objects': args.analyze_objects,
    'target_track_id': args.track_id,
    'tracker': args.tracker  # 
})

    # Batch mode
    if args.batch:
        process_batch(processor, args.batch, args.output_dir, analyze_objects=args.analyze_objects)
    # Single video
    elif args.input:
        output_path = args.output or os.path.join(args.output_dir, 'processed_output.mp4')
        process_single_video(processor, args.input, output_path, analyze_objects=args.analyze_objects)
    else:
        print("[ERROR] No input video provided.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()


# python main.py --input corrupted_video.mp4 --visualize-all --analyze-objects