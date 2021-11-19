from support import absPath

NAMES = {"11": "A1",
         "12": "A2",
         "13": "A3",
         "21": "B1",
         "22": "B2",
         "24": "B3",
         "32": "C1",
         "33": "C2",
         "34": "C3",
         "35": "C4",
         "43": "D1",
         "44": "D2",
         "45": "D3"}

def main(file_name: str):
	with open(absPath(file_name), "r") as f:
		content = f.readlines()
	
	new_content = list()

	for line in content:
		if line.startswith("Model"):
			new_content.append(line)
			continue
		c = line.split(";")
		c[0] = NAMES[c[0].split("_")[1]]
		new_content.append(";".join(c))

	with open(absPath(file_name), "w") as f:
		for line in new_content:
			f.write(line)

if __name__ == "__main__":
	main("../hte_z_mid.txt")