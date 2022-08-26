import numpy as np
from skimage.transform import resize
from collections import defaultdict
import pickle
from os.path import join
from pathlib import Path
import random
from tqdm import tqdm
from multiprocessing.pool import Pool
from functools import partial
import yaml
import zarr

# TODO: Add subject weights (map_weights, subject_weights, source_weight, ...?)
# TODO: Weights are a dict with name: weight for both class_weights and subject_weights

class Population:
    def __init__(self, population_dir, class_weights, length=None):
        self.store = population_dir
        self.class_weights = class_weights
        self.num_classes = len(self.class_weights)
        self.is_loaded = False
        self.length = length
        self.index = 0

    def compute(self, maps, downsize_factor):
        self.population_samples = defaultdict(dict)
        self.population_class_counts = defaultdict(dict)
        self.projection_vectors = {}
        self.map_shapes = {}

        for map_key, map in tqdm(maps.items()):
            print(map_key)
            map_array = np.array(map)
            map_downsampled, map_downsize_factor = _downsample_map(map_array, downsize_factor, self.num_classes)
            map_samples = _compute_samples(map_downsampled, self.num_classes)
            map_class_counts = _compute_class_counts(map_downsampled, self.num_classes)

            for class_id, samples in map_samples.items():
                self.population_samples[class_id][map_key] = samples

            for class_id, class_counts in map_class_counts.items():
                self.population_class_counts[class_id][map_key] = class_counts

            self.projection_vectors[map_key] = map_downsize_factor
            self.map_shapes[map_key] = map.shape

            data = {"population_class_counts": self.population_class_counts, "projection_vectors": self.projection_vectors, "map_shapes": self.map_shapes}

            with open(join(self.store, "data.pkl"), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.is_loaded = True

    def compute_parallel(self, maps, downsize_factor, processes):
        self.population_samples = defaultdict(dict)
        self.population_class_counts = defaultdict(dict)
        self.projection_vectors = {}
        self.map_shapes = {}

        pool = Pool(processes=processes)
        results = pool.map(partial(_compute_single, downsize_factor=downsize_factor, num_classes=self.num_classes, population_dir=self.store), maps.items())

        for map_key, map_samples, map_class_counts, map_downsize_factor, map_shape in tqdm(results):
            for class_id, samples in map_samples.items():
                self.population_samples[class_id][map_key] = samples

            for class_id, class_counts in map_class_counts.items():
                self.population_class_counts[class_id][map_key] = class_counts

            self.projection_vectors[map_key] = map_downsize_factor
            self.map_shapes[map_key] = map_shape

        data = {"population_class_counts": self.population_class_counts, "projection_vectors": self.projection_vectors, "map_shapes": self.map_shapes}

        with open(join(self.store, "data.pkl"), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.is_loaded = True

    def load(self):
        with open(join(self.store, "data.pkl"), 'rb') as handle:
            data = pickle.load(handle)

        self.population_class_counts = data["population_class_counts"]
        self.projection_vectors = data["projection_vectors"]
        self.map_shapes = data["map_shapes"]

        self.population_samples = defaultdict(dict)
        for key in self.population_class_counts.keys():
            for map_key in self.population_class_counts[key].keys():
                self.population_samples[key][map_key] = zarr.open(join(self.store, "population_samples", str(key), map_key), mode='r')

        self.is_loaded = True

    def apply_selection(self, selection):
        self.load()

        self.population_samples = _filter(self.population_samples, selection)
        self.population_class_counts = _filter(self.population_class_counts, selection)

        self.compute_statistics(throw_exception=True)

    def get_sample(self, class_weights=None):
        if not self.is_loaded:
            self.load()

        if class_weights is None:
            class_weights = self.class_weights

        class_weights = [class_weights[class_id] for class_id in range(len(class_weights)) if class_id in self.population_samples]
        class_id = random.choices(list(self.population_samples.keys()), weights=class_weights, k=1)[0]
        map_key = random.choices(list(self.population_samples[class_id].keys()), weights=list(self.population_class_counts[class_id].values()), k=1)[0]
        map = self.population_samples[class_id][map_key]
        sample = random.choice(map)
        sample = sample.astype(np.float64) * self.projection_vectors[map_key]
        sample = np.rint(sample).astype(np.int64)

        return sample, map_key, class_id

    def compute_statistics(self, throw_exception=False):
        if not self.is_loaded:
            self.load()

        counts = {}
        zero_count = False
        for key in self.population_class_counts.keys():
            class_counts = np.asarray(list(self.population_class_counts[key].values()))
            class_counts = np.sum(class_counts)
            counts[key] = class_counts
            print("{}: {}".format(key, class_counts))
            if class_counts == 0:
                zero_count = True

        sum = np.sum(np.asarray(list(counts.values())))
        counts_ratio = {key: counts[key] / sum for key in counts.keys()}
        print(counts_ratio)

        del counts[0]
        sum = np.sum(np.asarray(list(counts.values())))
        counts_ratio = {key: counts[key] / sum for key in counts.keys()}
        print(counts_ratio)

        if throw_exception and zero_count:
            raise RuntimeError("One or more classes are not present in the population.")


    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.get_sample()
        return sample

    def __next__(self):
        if self.index < self.length:
            sample = self.__getitem__(-1)
            self.index += 1
            return sample
        else:
            raise StopIteration


def _compute_single(item, downsize_factor, num_classes, population_dir):
    map_key, map = item
    print("Key: ", map_key)
    map_array = np.array(map)
    map_downsampled, map_downsize_factor = _downsample_map(map_array, downsize_factor, num_classes)
    map_samples = _compute_samples(map_downsampled, num_classes)
    map_class_counts = _compute_class_counts(map_downsampled, num_classes)
    for class_id, samples in map_samples.items():
        Path(join(population_dir, "population_samples", str(class_id))).mkdir(parents=True, exist_ok=True)
        samples_zarr = zarr.array(samples)
        print("{}: {}".format(map_key, join(population_dir, "population_samples", str(class_id), map_key)))
        zarr.save(join(population_dir, "population_samples", str(class_id), map_key), samples_zarr)
        samples_zarr = zarr.open(join(population_dir, "population_samples", str(class_id), map_key), mode='r')
        map_samples[class_id] = samples_zarr
    return map_key, map_samples, map_class_counts, map_downsize_factor, map.shape


def _downsample_map(map, downsize_factor, num_classes):
    map = np.asarray(map)
    original_shape = np.asarray(map.shape)
    if downsize_factor is not None:
        downsized_shape = (original_shape / downsize_factor).astype(int)
        real_downsize_factor = original_shape / downsized_shape
        downsampled_map = _downsample(map, downsized_shape, num_classes)
        # downsampled_map = _smooth_seg_resize(map, downsized_shape, labels=list(range(num_classes+1)))
    else:
        downsampled_map = _one_hot_encode(map, map.shape, num_classes)
        real_downsize_factor = 1
    return downsampled_map, real_downsize_factor


def _downsample(seg, target_shape, num_classes, order=0):
    downsampled = np.zeros((num_classes, *target_shape), dtype=np.uint8)

    for class_id in range(num_classes):
        mask = seg == class_id
        downsampled[class_id] = resize(mask, target_shape, order, anti_aliasing=False)

    return downsampled


def _one_hot_encode(seg, target_shape, num_classes):
    one_hot_ecoded = np.zeros((num_classes, *target_shape), dtype=np.uint8)

    for class_id in range(num_classes):
        one_hot_ecoded[class_id] = seg == class_id

    return one_hot_ecoded


def _smooth_seg_resize(seg: np.ndarray, target_shape, order=1, labels=None, continuous=True, progressbar=False) -> np.ndarray:
    """Order should be between 1-3. The higher the smoother, but also longer."""
    reshaped = np.zeros(target_shape, dtype=seg.dtype)
    if labels is None:
        if continuous:
            labels = list(range(np.max(seg) + 1))
        else:
            labels = np.unique(seg)

    for i, label in enumerate(tqdm(labels, desc="Smooth Resampling", disable=not progressbar)):
        mask = seg == label
        reshaped_multihot = resize(mask.astype(float), target_shape, order, mode="edge", clip=True, anti_aliasing=False)
        reshaped[reshaped_multihot >= 0.5] = label
    return reshaped


def _compute_samples(map, num_classes):
    samples = {}

    for class_id in range(num_classes):
        class_samples = np.argwhere(map[class_id] == 1)
        if len(class_samples) > 0:
            samples[class_id] = class_samples

    return samples


def _compute_class_counts(map, num_classes):
    class_counts = {}

    for class_id in range(num_classes):
        class_count = np.count_nonzero(map[class_id])
        if class_count > 0:
            class_counts[class_id] = class_count

    return class_counts


def _filter(data_dict, selection):
    keys = list(data_dict.keys())
    for key in keys:
        map_keys = list(data_dict[key].keys())
        for map_key in map_keys:
            if map_key not in selection:
                del data_dict[key][map_key]
        if len(data_dict[key]) == 0:
            del data_dict[key]
    return data_dict


def load_filepaths(dataset_path, masks="/masks_2/"):
    names = []
    for fold in range(5):
        with open(join(dataset_path, "Splits", "Split{}.txt".format(fold+1))) as f:
            fold_names = f.readlines()
        fold_names = [join(dataset_path, name[1:-1]) for name in fold_names]
        fold_names = [name.replace("/imgs/", masks) for name in fold_names]
        names.append(fold_names)
    return names


def load_subjects(filepaths):
    subjects = {}
    for filepath in filepaths:
        image = zarr.open(filepath, mode="r")
        seg = zarr.open(filepath, mode="r")
        subject = {aug.IMAGE: image, aug.SEG: seg}
        subjects[filepath] = subject
    return subjects


if __name__ == '__main__':
    from slicer import slicer
    import augmentify as aug

    # population_dir = "/home/k539i/Documents/datasets/preprocessed/AGGC2022"
    # dataset_path = "/home/k539i/Documents/datasets/preprocessed/AGGC2022/Subset1"
    # names = ["Subset1_Train_1", "Subset1_Train_2", "Subset1_Train_3", "Subset1_Train_4", "Subset1_Train_5", "Subset1_Train_6", "Subset1_Train_7", "Subset1_Train_8", "Subset1_Train_9"]
    #
    # # maps = [zarr.open(join(dataset_path, "masks", name + ".zarr"), mode='r') for name in names]
    # maps = {name: zarr.open(join(dataset_path, "masks", name + ".zarr"), mode='r') for name in names}

    with open("storage_worker.yml", "r") as stream:
        storage = yaml.safe_load(stream)

    population_dir = storage["storage"]["population_dir"]
    dataset_path = storage["storage"]["dataset_path"]
    filepaths = load_filepaths(dataset_path, masks="/masks_2/")
    filepaths = np.concatenate(filepaths).tolist()

    maps = {filepath.split("/")[-1]: zarr.open(filepath, mode='r') for filepath in filepaths}
    # maps = {"Subset1_Train_1.zarr": zarr.open("/home/k539i/Documents/datasets/preprocessed/AGGC2022/dataset/masks/Subset1_Train_1.zarr/", mode='r')}

    population = Population(population_dir=population_dir, class_weights=(1, 1, 1, 1, 1, 1))
    population.compute_parallel(maps, 10, processes=30)
    # population.compute_parallel(maps, None, processes=5)
    population.compute_statistics()
    # population.load()
    # population.save("/dkfz/cluster/gpu/data/OE0441/k539i/original/AGGC2022/population_all")

    # correctly_sampled = []
    # for _ in tqdm(range(100000)):
    #     sample, map_key, class_id = population.get_sample()
    #     sample_point = maps[map_key][slicer(maps[map_key], sample)]
    #     # print("class_id: {}, is_sampled: {}".format(class_id, sample_point == class_id))
    #     correctly_sampled.append(sample_point == class_id)
    # correctly_sampled = np.sum(correctly_sampled) / len(correctly_sampled)
    # print("correctly_sampled: ", correctly_sampled)
    #
    # with open('correctly_sampled.txt', 'w') as f:
    #     f.write(correctly_sampled)
