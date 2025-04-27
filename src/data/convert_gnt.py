import os, struct
from PIL import Image
from tqdm import tqdm

def convert_gnt_folder(gnt_folder, out_folder):
    for root, _, files in os.walk(gnt_folder):
        for fname in tqdm(files, desc=f"Converting {gnt_folder}"):
            if not fname.endswith('.gnt'):
                continue
            path = os.path.join(root, fname)
            with open(path, 'rb') as f:
                while True:
                    header = f.read(10)
                    if not header:
                        break
                    _, label, w, h = struct.unpack('<IHHH', header)
                    img_data = f.read(w*h)
                    img = Image.frombytes('L', (w, h), img_data)

                    label_hex = f"{label:04x}"
                    class_dir = os.path.join(out_folder, label_hex)
                    os.makedirs(class_dir, exist_ok=True)

                    stamp = f.tell()
                    out_name = f"{label_hex}_{stamp}.png"
                    img.save(os.path.join(class_dir, out_name))

if __name__=="__main__":
    mappings = {
        "data/CASIA/raw/train1": "data/CASIA/images/train",
        "data/CASIA/raw/train2": "data/CASIA/images/train",
        "data/CASIA/raw/test":   "data/CASIA/images/test",
    }
    # clear old outputs (if any)
    for dst in mappings.values():
        if os.path.exists(dst):
            os.system(f"rm -rf {dst}")
    for src, dst in mappings.items():
        print(f"→ Converting {src} → {dst}")
        convert_gnt_folder(src, dst)