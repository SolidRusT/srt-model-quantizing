import json
import os
import shutil
import torch
from collections import defaultdict
from safetensors.torch import load_file, save_file
from tqdm import tqdm

class Config:
    COPY_ADD_DATA_DEFAULT = True
    DELETE_OLD_DEFAULT = False
    SOURCE_FOLDER_DEFAULT = ''
    DEST_FOLDER_DEFAULT = ''
    FILE_SIZE_DIFFERENCE_THRESHOLD = 0.01
    MODEL_NAME = 'model'
    FILE_EXTENSION = '.safetensors'
    INDEX_FILE_NAME = 'model.safetensors.index.json'
    CONVERTED_FORMAT = 'pt'

class FileConverter:
    def __init__(self, source_folder, dest_folder, delete_old):
        self.source_folder = source_folder
        self.dest_folder = dest_folder
        self.delete_old = delete_old

    @staticmethod
    def shared_pointers(tensors):
        ptrs = defaultdict(list)
        for k, v in tensors.items():
            ptrs[v.data_ptr()].append(k)
        return [names for names in ptrs.values() if len(names) > 1]

    @staticmethod
    def check_file_size(sf_filename, pt_filename):
        sf_size = os.stat(sf_filename).st_size
        pt_size = os.stat(pt_filename).st_size
        if (sf_size - pt_size) / pt_size > Config.FILE_SIZE_DIFFERENCE_THRESHOLD:
            raise RuntimeError(f"File size difference exceeds 1% between {sf_filename} and {pt_filename}")

    def convert_file(self, pt_filename, sf_filename, copy_add_data=Config.COPY_ADD_DATA_DEFAULT):
        loaded = torch.load(pt_filename, map_location="cpu")
        loaded = loaded.get("state_dict", loaded)
        shared = self.shared_pointers(loaded)

        for shared_weights in shared:
            for name in shared_weights[1:]:
                loaded.pop(name)

        loaded = {k: v.contiguous().half() for k, v in loaded.items()}

        os.makedirs(self.dest_folder, exist_ok=True)
        save_file(loaded, sf_filename, metadata={"format": Config.CONVERTED_FORMAT})
        self.check_file_size(sf_filename, pt_filename)
        if copy_add_data:
            self.copy_additional_files(self.source_folder, self.dest_folder)

        reloaded = load_file(sf_filename)
        for k, v in loaded.items():
            if not torch.equal(v, reloaded[k]):
                raise RuntimeError(f"Mismatch in tensors for key {k}.")

    @staticmethod
    def rename(pt_filename):
        return pt_filename.replace("pytorch_model", Config.MODEL_NAME).replace(".bin", Config.FILE_EXTENSION)

    @staticmethod
    def copy_additional_files(source_folder, dest_folder):
        for file in os.listdir(source_folder):
            file_path = os.path.join(source_folder, file)
            if os.path.isfile(file_path) and not (file.endswith('.bin') or file.endswith('.py')):
                shutil.copy(file_path, dest_folder)

    def find_index_file(self):
        for file in os.listdir(self.source_folder):
            if file.endswith('.bin.index.json'):
                return file
        return None

    def convert_files(self):
        index_file = self.find_index_file()
        if not index_file:
            raise RuntimeError("Index file not found. Please ensure the correct folder is specified.")

        index_file = os.path.join(self.source_folder, index_file)
        with open(index_file) as f:
            index_data = json.load(f)

        for pt_filename in tqdm(set(index_data["weight_map"].values())):
            full_pt_filename = os.path.join(self.source_folder, pt_filename)
            sf_filename = os.path.join(self.dest_folder, self.rename(pt_filename))
            self.convert_file(full_pt_filename, sf_filename, copy_add_data=False)
            if self.delete_old:
                os.remove(full_pt_filename)

        self.copy_additional_files(self.source_folder, self.dest_folder)
        
        index_path = os.path.join(self.dest_folder, Config.INDEX_FILE_NAME)
        with open(index_path, "w") as f:
            new_map = {k: self.rename(v) for k, v in index_data["weight_map"].items()}
            json.dump({**index_data, "weight_map": new_map}, f, indent=4)

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    source_folder = input("Source folder for PyTorch files (leave blank for script's directory): ").strip() or script_dir
    dest_folder = input("Destination folder for SafeTensors files (leave blank for default): ").strip()

    if not dest_folder:
        model_name = os.path.basename(os.path.normpath(source_folder))
        dest_folder = os.path.join(source_folder, model_name + "_safetensors")

    delete_old = input("Delete old PyTorch files? (Y/N): ").strip().upper() == 'Y'

    converter = FileConverter(source_folder, dest_folder, delete_old)

    if "pytorch_model.bin" in os.listdir(source_folder):
        converter.convert_file(os.path.join(source_folder, "pytorch_model.bin"), os.path.join(dest_folder, "model.safetensors"))
        if delete_old:
            os.remove(os.path.join(source_folder, "pytorch_model.bin"))
    else:
        converter.convert_files()

if __name__ == "__main__":
    main()
