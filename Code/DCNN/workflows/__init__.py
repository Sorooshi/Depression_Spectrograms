import sys
import pathlib

parentdir = str(pathlib.Path(__file__).parent.parent.parent.resolve())
print(parentdir)
sys.path.append(parentdir)
print(sys.path)