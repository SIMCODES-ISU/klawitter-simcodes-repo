def fibonacci(i):
	seq = [0, 1]
	
	while len(seq) <= i:
		seq.append(seq[-2] + seq[-1])

	return seq[i]

