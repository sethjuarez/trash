import os
import json

class TrashData:
    def __init__(self) -> None:
        self.classes = []
        self.data = []

    def load(self, path: str) -> None:
        classes_path = os.path.join(path, 'classes.json')
        dataset_path = os.path.join(path, 'new_dataset.json')

        if not os.path.exists(path):
            raise Exception('Invalid path')

        with open(classes_path) as f:
            self.classes = json.load(f)

        with open(dataset_path) as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __getitem__(self, i):
        if i < 0 and i >= len(self.data):
            raise Exception('Out of range')
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def class_count(self) -> int:
        return len(self.data)

if __name__ == '__main__':
    c = TrashData()
    c.load('data/dataset_surfrider_cleaned')

    for i in range(len(c)):
        print(c[i])

    print(c.classes)
    print('--------------------')
    print(c.data)