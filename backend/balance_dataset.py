# Balance Dataset and Remove Specific Classes
# This script loads your original pickle, balances classes, and removes unwanted classes

import pickle
import random
from collections import defaultdict

def balance_and_filter_dataset(
    input_file='new_data.pickle',
    output_file='data_test.pickle',
    max_per_class=612,
    exclude_classes=['nothing']
):
    """
    Create a balanced dataset from original pickle file
    
    Args:
        input_file: Path to original pickle file
        output_file: Path to save balanced pickle file
        max_per_class: Maximum samples per class
        exclude_classes: List of class names to exclude (e.g., ['nothing', 'del'])
    """
    print(f"Loading dataset from: {input_file}")
    
    # Load original pickle
    with open(input_file, 'rb') as f:
        orig = pickle.load(f)
    
    data = orig['data']
    labels = orig['labels']
    
    print(f"\nOriginal dataset:")
    print(f"  Total samples: {len(data)}")
    print(f"  Total classes: {len(set(labels))}")
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        class_indices[lbl].append(idx)
    
    # Show original class distribution
    print(f"\nOriginal class distribution:")
    for lbl in sorted(class_indices.keys()):
        print(f"  {lbl}: {len(class_indices[lbl])} samples")
    
    # Balance and filter
    balanced_data = []
    balanced_labels = []
    
    print(f"\n{'='*60}")
    print(f"Balancing dataset (max {max_per_class} per class)...")
    print(f"Excluding classes: {exclude_classes}")
    print(f"{'='*60}\n")
    
    for lbl, idxs in sorted(class_indices.items()):
        # Skip excluded classes
        if lbl in exclude_classes:
            print(f"✗ Skipping class '{lbl}': {len(idxs)} samples (excluded)")
            continue
        
        # Sample up to max_per_class
        num_to_sample = min(len(idxs), max_per_class)
        sampled = random.sample(idxs, num_to_sample)
        
        for idx in sampled:
            balanced_data.append(data[idx])
            balanced_labels.append(lbl)
        
        print(f"✓ Class '{lbl}': Selected {num_to_sample} samples")
    
    # Save balanced dataset
    balanced_dict = {
        'data': balanced_data,
        'labels': balanced_labels
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(balanced_dict, f)
    
    print(f"\n{'='*60}")
    print(f"Balanced dataset created!")
    print(f"{'='*60}")
    print(f"Output file: {output_file}")
    print(f"Total samples: {len(balanced_data)}")
    print(f"Total classes: {len(set(balanced_labels))}")
    print(f"Classes: {sorted(set(balanced_labels))}")
    
    # Show final class distribution
    final_counts = defaultdict(int)
    for lbl in balanced_labels:
        final_counts[lbl] += 1
    
    print(f"\nFinal class distribution:")
    for lbl in sorted(final_counts.keys()):
        print(f"  {lbl}: {final_counts[lbl]} samples")
    
    print(f"\n✓ Ready for training! Use: python train_asl_model.py")
    print(f"  (Make sure to update the pickle file path in train script if needed)")

def main():
    # Configure your settings here
    INPUT_FILE = 'new_data.pickle'              # Your original pickle file
    OUTPUT_FILE = 'data_test.pickle'  # New balanced file
    MAX_PER_CLASS = 602                # Max samples per class
    EXCLUDE_CLASSES = ['nothing']           # Classes to remove
    
    # You can also exclude other classes if needed:
    # EXCLUDE_CLASSES = ['nothing', 'del', 'space']
    
    balance_and_filter_dataset(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        max_per_class=MAX_PER_CLASS,
        exclude_classes=EXCLUDE_CLASSES
    )

if __name__ == "__main__":
    main()
