import glob


def count_lines(pattern):
    lines = 0
    for p in glob.glob(pattern):
        with open(p) as f:
            for line in f:
                line.strip()
                if line and line[0] not in ["#", "\"", "'", "/"]:
                    lines += 1

    return lines


print("Python lines:", count_lines("torch_radon/*.py"))
print("Test lines (python):", count_lines("tests/*.py"))
print("Examples lines (python):", count_lines("examples/*.py"))

print("CUDA lines:", count_lines("src/*.cu"))
print("C++ lines:", count_lines("src/*.cpp") + count_lines("include/*.h"))
