from datasets import load_dataset, load_from_disk

full = load_from_disk(r"C:\junha\Datasets\BookCorpus")
splits = full.train_test_split(test_size=0.1, seed=42)
train_map = splits["train"]
val_map = splits["test"]

train_map.save_to_disk(r"C:\junha\Datasets\BookCorpus\train")
val_map.  save_to_disk(r"C:\junha\Datasets\BookCorpus\val")


print("Cache_dir에 bookcorpus 데이터 다운 완료. streaming=True이어도 네트워크 연결 X")
