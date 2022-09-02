import glob
import os
import numpy as np
if __name__ == '__main__':
	root = '.'
	writeLabelNamePath = root + '/LabelName.txt'
	writeTrainPath = root + '/train.csv'
	writeTestPath = root + '/test.csv'
	writeValPath = root + '/val.csv'

	fwLN = open(writeLabelNamePath, "w", encoding='UTF-8')

	dirLS = os.listdir(root)
	# =========================
	# Arrange
	Temp = []
	for idx, dirName in enumerate(dirLS):
		dirPath = root + '/' + dirName
		if(os.path.isdir(dirPath)):
			Temp.append(dirName)
	dirLS = Temp
	# =========================

	DataSet = []
	for idx, dirName in enumerate(dirLS):
		fwLN.write("%s, %s" % (str(idx), dirName))
		dirPath = root + '/' + dirName
		for filePath in glob.glob(dirPath + '/*.jpg'):
			fileName = os.path.basename(filePath)
			DataSet.append([dirName + "/" + fileName, idx])

		fwLN.write("\n")

	DataSet = np.asarray(DataSet, dtype=str)

	# Random
	num_example = DataSet.shape[0]
	arr = np.arange(num_example)
	np.random.shuffle(arr)
	DataSet = DataSet[arr]

	# Split Train 80%(Train 80%/Val 20%), Test 20%
	# Train Test
	s = np.int32(num_example * 0.8)
	TrainSet = DataSet[:s]
	TestSet = DataSet[s:]

	# Train Val
	num_example = TrainSet.shape[0]
	s = np.int32(num_example * 0.8)
	ValSet = TrainSet[s:]
	TrainSet = TrainSet[:s]

	np.savetxt(writeTrainPath, TrainSet, delimiter=",", fmt ='%s')
	np.savetxt(writeTestPath, TestSet, delimiter=",", fmt ='%s')
	np.savetxt(writeValPath, ValSet, delimiter=",", fmt ='%s')

	fwLN.close()
