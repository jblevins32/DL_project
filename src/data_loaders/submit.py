import os
import zipfile
import glob

_A1_FILES = [
    "models/softmax_regression.py",
    "models/__init__.py",
    "models/two_layer_nn.py",
    "models/_base_network.py",
    "optimizer/sgd.py",
    "optimizer/__init__.py",
    "optimizer/_base_optimizer.py",
    "data/data_processing.py",
]

_A2_1_FILES = [
    "modules/conv_classifier.py",
    "modules/__init__.py",
    "modules/convolution.py",
    "modules/linear.py",
    "modules/max_pool.py",
    "modules/relu.py",
    "modules/softmax_ce.py",
    "optimizer/sgd.py",
    "optimizer/__init__.py",
    "optimizer/_base_optimizer.py",
]

_A2_2_FILES = [
    "models/__init__.py",
    "models/cnn.py",
    "models/my_model.py",
    "models/resnet.py",
    "models/twolayer.py",
    "losses/__init__.py",
    "losses/focal_loss.py",
    "checkpoints/mymodel.pth",
    "checkpoints/twolayernet.pth",
    "checkpoints/vanillacnn.pth",
    "configs/*.yaml",
]


def make_a1_submission(assignment_path):
    _make_submission(assignment_path, _A1_FILES, "assignment_1_submission")


def make_a2_1_submission(assignment_path):
    _make_submission(assignment_path, _A2_1_FILES, "assignment_2_1_submission")


def make_a2_2_submission(assignment_path):
    _make_submission(assignment_path, _A2_2_FILES, "assignment_2_2_submission")


def try_write_file(path, filename, zf):
    if not os.path.isfile(path):
        raise ValueError('Could not find file "%s"' % filename)
    zf.write(path, filename)


def _make_submission(assignment_path, file_list, assignment_name):
    zip_path = os.path.join(assignment_path, (assignment_name + ".zip"))
    print("Writing zip file to: ", zip_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for filename in file_list:
            if "*" in filename:
                file_list2 = glob.glob(os.path.join(assignment_path, filename))
                for filename2 in file_list2:
                    in_path = os.path.join(assignment_path, filename2)
                    try_write_file(
                        in_path, os.path.join("config", os.path.basename(filename2)), zf
                    )
            else:
                in_path = os.path.join(assignment_path, filename)
                try_write_file(in_path, filename, zf)
