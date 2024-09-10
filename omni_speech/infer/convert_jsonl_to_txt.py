import sys
import json

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    data = fin.readlines()
    for line in data:
        item = json.loads(line)
        prediction_units = item["prediction_units"]
        if prediction_units != "":
            fout.write(prediction_units + "\n")
        else:
            fout.write("0\n")
