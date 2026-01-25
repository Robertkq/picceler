import lit.formats
import os

config.name = "Picceler"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']

build_dir = os.path.abspath("../build/")

config.substitutions.append(('%picceler-opt', os.path.join(build_dir, 'picceler-opt')))