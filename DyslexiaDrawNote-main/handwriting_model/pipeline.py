# handwriting_model/pipeline.py

import sys
import os
import cv2
import uuid
import json



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.word_detector import prepare_img, detect, sort_multiline
from server.char_segmentor import segment_char_images
from handwriting_model.emnist_model import predict_emnist
from handwriting_model.reversal_model import predict_reversal


def run_pipeline(image_path: str, output_json: str = None):
    # 1) Load & normalize page
    img_raw = cv2.imread(image_path)
    

    img_gray = prepare_img(img_raw, height=800)

    # 2) Word detection + sorting
    detections = detect(img_gray, kernel_size=151, sigma=6, theta=10, min_area=400)
    print(f"[DEBUG] Detected {len(detections)} word candidates")


    words = sort_multiline(detections)

    results = []
    # 3) For each word → segment chars → run both models
    for line_idx, line in enumerate(words):
        for word_idx, det in enumerate(line):
            x,y,w,h = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
            word_img = img_gray[y:y+h, x:x+w]
            chars = segment_char_images(word_img)
            

            debug_dir = f"server/uploads/chars/word_{word_idx}"
            os.makedirs(debug_dir, exist_ok=True)

            for i, char_img in enumerate(chars):
                fname = f"{uuid.uuid4().hex[:8]}.png"
                cv2.imwrite(os.path.join(debug_dir, fname), char_img)


            word_str = ""
            flips = []
            for char_img in chars:
                # EMNIST expects 28×28
                em = cv2.resize(char_img, (28,28), interpolation=cv2.INTER_AREA)
                lbl, p_em = predict_emnist(em)

                # Binary expects 32×32
                bi = cv2.resize(char_img, (32,32), interpolation=cv2.INTER_AREA)
                p_bi = predict_reversal(bi)

                # choose flip with higher confidence
                flip_confidence = max(p_em, p_bi)
                flip_flag = int(flip_confidence == p_bi)

                word_str += lbl
                flips.append(flip_flag)

            results.append({
                "line": line_idx,
                "word": word_idx,
                "text": word_str,
                "flips": flips
            })

    # 4) Optional: save to JSON
    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]
    out = run_pipeline(img_path)
    print(json.dumps(out))
