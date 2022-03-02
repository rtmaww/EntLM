
class Result():
    sample_num = None
    round_num = None
    f1 = None
    CRF_f1 = None

    def is_valid(self):
        for name, value in vars(self).items():
            if value is None:
                return False
        return True



log_name = "test.log"

with open(log_name, "r") as f:
    lines = f.readlines()

result_list = []
result = Result()
for idx, line in enumerate(lines):
    if "-----Training with file" in line:
        # if result.f1 is not None:
        #     result_list.append(result)
        result = Result()
        result.sample_num = line.strip().split("/")[-1][0]
        result.round_num = line.strip().split()[-1][0]
        print(result.sample_num, result.round_num)

    if "Finish training, best metric:" in line and idx+1 < len(lines):
        metric = eval(lines[idx+1])
        # print(metric)
        result.f1 = metric["overall_f1"]
        print(result.f1)

    if "Decoding with CRF:" in line and idx+2 < len(lines):
        # metric = eval(lines[idx+2])
        # # print(metric)
        # result.CRF_f1 = metric["overall_f1"]
        i = idx + 1
        while i < len(lines):
            if "overall_f1" in lines[i]:
                result.CRF_f1 = lines[i].split("overall_f1: ")[-1].split(",")[0]
                print(result.CRF_f1)
                break
            i += 1

        if result.f1 is not None:
            result_list.append(result)


from collections import defaultdict
sample_num = -1
outputs = defaultdict(dict)
crf_outputs = defaultdict(dict)
for result in result_list:
    print(result)
    outputs[int(result.sample_num)][int(result.round_num)] = round(float(result.f1) * 100,4)
    crf_outputs[int(result.sample_num)][int(result.round_num)] = round(float(result.CRF_f1) * 100,4)




round = 4
sample_num = 4
if "conll" in log_name:
    sample_num = 4

print("F1:")

for i in range(1,1+round):
    to_print = ""
    for s in range(1, sample_num+1):
        to_print += str(outputs[s][i])+"\t"
    print(to_print[:-1])

print()
for i in range(1, 1 + round):
    to_print = ""
    for s in range(1, sample_num+1):
        to_print += str(crf_outputs[s][i])+"\t"
    print(to_print[:-1])
