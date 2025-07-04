import sys
import os
import cv2
import json

# Add project root to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.word_detector import prepare_img, detect, sort_multiline ,_cluster_lines
from server.char_segmentor import segment_char_images
from handwriting_model.emnist_model import predict_emnist
from handwriting_model.reversal_model import predict_reversal


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def pipeline(input_path, output_base="output"):
    # Prepare directories
    raw_dir = os.path.join(output_base, 'raw')
    lines_dir = os.path.join(output_base, 'lines')
    words_dir = os.path.join(output_base, 'words')
    chars_dir = os.path.join(output_base, 'chars')
    ensure_dir(raw_dir)
    ensure_dir(lines_dir)
    ensure_dir(words_dir)
    ensure_dir(chars_dir)
    


    # Step 1: Copy raw image
    raw_dest = os.path.join(raw_dir, os.path.basename(input_path))
    cv2.imwrite(raw_dest, cv2.imread(input_path))

    # Step 2: Prepare image and detect words across page
    img_raw = cv2.imread(raw_dest)
    img_gray = prepare_img(img_raw, height=800)

    print(f"[DEBUG] Raw image shape: {img_raw.shape}")
    print(f"[DEBUG] Prepared grayscale shape: {img_gray.shape}")

    detections = detect(img_gray, kernel_size=151,
    sigma=6,
    theta=10,
    min_area=400)
    line_groups = _cluster_lines(detections)  # returns List[List[DetectorRes]]
    lines = [sort_multiline(line)[0] for line in line_groups if len(line) > 0]

    print(f"[DEBUG] Found {len(detections)} total word-like regions")
    for i, det in enumerate(detections):
        print(f"  - Detection {i}: bbox=({det.bbox.x}, {det.bbox.y}, {det.bbox.w}, {det.bbox.h})")

    print(f"[DEBUG] Organized into {len(lines)} lines")
    for i, line in enumerate(lines):
        print(f"  - Line {i} has {len(line)} words")



    results = []
    # Loop over lines and words
    if not lines:
        print("[ERROR] No lines or words detected. Please check your input image or preprocessing parameters.")
        return []

    for line_idx, line in enumerate(lines):
        line_dir = os.path.join(lines_dir, f"line_{line_idx}")
        ensure_dir(line_dir)
        for det in line:
            x, y, w, h = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
            crop = img_gray[y:y+h, x:x+w]
            # Save word image
            word_idx = next(i for i, c in enumerate(line) if c.bbox == det.bbox)

            # ✅ safe compare
            word_dir = os.path.join(words_dir, f"line_{line_idx}_word_{word_idx}")
            ensure_dir(word_dir)
            word_img_path = os.path.join(word_dir, f"word_{word_idx}.png")
            cv2.imwrite(word_img_path, crop)

            # Step 3: Segment characters in word
            chars = segment_char_images(crop)
            char_results = []
            for char_idx, char in enumerate(chars):
                # Save char image for debug
                char_dir = os.path.join(chars_dir, f"line_{line_idx}_word_{word_idx}")
                ensure_dir(char_dir)
                char_path = os.path.join(char_dir, f"char_{char_idx}.png")
                cv2.imwrite(char_path, char)

                # Step 4: Predict char and flip
                em_lbl, em_prob = predict_emnist(char)
                bi_prob = predict_reversal(char)
                char_results.append({
                    'char_idx': char_idx,
                    'label': em_lbl,
                    'flip_emnist': em_prob,
                    'flip_binary': bi_prob
                })

            results.append({
                'line': line_idx,
                'word': word_idx,
                'text': ''.join([c['label'] for c in char_results]),
                'flips': [int(c['flip_binary'] > c['flip_emnist']) for c in char_results],
                'chars': char_results
            })

    # Save overall results
    with open(os.path.join(output_base, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pipeline_full.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    res = pipeline(img_path)

    for item in res:
        print(f"\nLine {item['line']} | Word {item['word']} -> '{item['text']}'")
        for char_info in item['chars']:
            label = char_info['label']
            flip_emnist = char_info['flip_emnist']
            flip_binary = char_info['flip_binary']
            is_flipped = "Yes" if flip_binary > 0.5 else "No"
            print(f"  - Char[{char_info['char_idx']}] = '{label}' | Flipped? {is_flipped} | EMNIST: {flip_emnist:.2f} | Binary: {flip_binary:.2f}")

