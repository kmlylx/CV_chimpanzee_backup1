from torchvision import datasets
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, find_classes, has_file_allowed_extension
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import os



class MyImageFolder(DatasetFolder):
    def __init__(
        self, root: str, loader: Callable[[str], Any] = default_loader, extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, is_valid_file: Optional[Callable[[str], bool]] = None
    ):

        super().__init__(root, loader, extensions, transform=transform, target_transform=target_transform)



    @staticmethod
    def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        
        print("Override 'make_dataset'")
        
        # Because one class validation directory is empty and DatasetFolder will ignore this class by default, which will distort the class_to_idx, we override the make dataset function so that it will consider all classes.
        # The only change is in the last line. We don't raise an error if this class doesn't contain any file.
        
        # At first, I thought this might cause the error. But then I deleted the empty directory and the error persisted.
    
        print(directory)
        directory = os.path.expanduser(directory)
        print(directory)

        if class_to_idx is None:
            _, class_to_idx = find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
                
    #         raise FileNotFoundError(msg)

        return instances


